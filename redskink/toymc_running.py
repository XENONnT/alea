from tqdm import tqdm
from subprocess import call
from time import gmtime, strftime, sleep
import argparse
from json import dumps, loads
from string import Formatter
import numpy as np
from copy import deepcopy
from datetime import datetime
from os.path import splitext
import pkg_resources
import warnings
import inspect
from itertools import product
import re
from glob import glob
import h5py
from pydoc import locate  # to lookup inferenceObject class
import os
from inference_interface import numpy_to_toyfile
import binference
import copy
import mergedeep
import logging
logging.basicConfig(level=logging.INFO)


def sbatch_template(queue, logfile, execution_time, ram_roof,
                    singularity_container):
    template_script = """#!/bin/sh

#SBATCH --account=pi-lgrandi
#SBATCH --ntasks=1
#SBATCH --partition={queue}
#SBATCH --qos={queue}
#SBATCH --output={logfile}
#SBATCH --error={logfile}
#SBATCH --time={execution_time}
#SBATCH --mem-per-cpu={ram_roof}

# this bash-script receives the same inputs as the run_toymc.py script


toymc_script=$(python -c 'import pkg_resources; print(pkg_resources.resource_filename("binference","/scripts/run_toymc.py"))')
execution_cmd="python $toymc_script $@"
echo $execution_cmd
echo "-------------------"

echo "loading singularity"
module load singularity

echo "executing command"
singularity exec --bind /project2 --bind /dali --bind /home {singularity_container} $execution_cmd
"""
    return template_script.format(queue=queue,
                                  execution_time=execution_time,
                                  logfile=logfile,
                                  ram_roof=ram_roof,
                                  singularity_container=singularity_container)


def create_sbatch_file(
    filename="runpy.sbatch",
    queue="xenon1t",
    logfile="logfile.log",
    execution_time=60,
    ram_roof=2000,
    singularity_container="/project2/lgrandi/xenonnt/singularity-images/xenonnt-development.simg"
):
    # getting bash-script template and set computing stuff
    bash_text = sbatch_template(queue=queue,
                                logfile=logfile,
                                execution_time=execution_time,
                                ram_roof=ram_roof,
                                singularity_container=singularity_container)

    # write bash-script to file
    with open(filename, "w") as f:
        f.write(bash_text)

    #  if os.path.exists(filename):
    #      print("batch-file", filename, "created.")
    #  else:
    #      print("batch-file", filename, "NOT created.")


def add_imc(fpat):
    if "i_batch" not in [tup[1] for tup in Formatter().parse(fpat)]:
        fpat_split = splitext(fpat)
        return fpat_split[0] + "_{i_batch:d}" + fpat_split[1]
    else:
        return fpat


def run_toymcs(
    n_mc=10,
    i_batch=0,  #indexes batch-dependent values
    wimp_mass=50,
    signal_expectation=None,  # no of evts.
    signal_component_name="signal",  #change this is you're doing inference on the rate of a source not named signal
    generate_args={},
    livetime=1.,  #ty
    extra_args=[{}],
    extra_args_all={},
    guess={},
    compute_confidence_interval=False,
    limit_threshold=None,
    threshold_key=None,
    inference_object_args={},
    refit_if_first_not_best=False,
    propagate_guess=True,
    return_expectations=False,
    inference_class_name="binference.likelihoods.ll_nt.InferenceObject",
    toydata_file=None,
    toydata_mode="none",
):
    """
    run toyMCs and return best-fit likelihoods,
    best-fit parameters, confidence intervals

    input args:
    -----------
        n_mc: int, number of toyMCs, defaults to 10
        i_batch: indexes batch-dependent values, defaults to 0
        wimp_mass: mass of signal in GeV, defaults to 50
        signal_expectation: number of events expected, defaults to None
            if not None -- will set the signal_rate_multiplier in the
            toyMC to have this signal expectation.
        signal_component_name: defaults to 'signal'
            change this if you're doing inference on the rate of a
            source not named signal
        generate_args: kwargs passed to simulate. Possible kwargs are the
            rate, signal and shape parameters, defaults to {}
        livetime: exposure in t*y, defaults to 1.
        extra_args: array of dicts, defaults to [{}]
            the function will fit each toyMC with each extra_arg
            (so e.g. fit for signal=0, signal="free", signal="true")
        extra_args_all: dict of full extra_args passed to each fit,
            defaults to {}
            (overridden by the per-fit extra_args)
        guess: dict passed to fitter as guess, defaults to {}
        compute_confidence_interval: defaults to False
            if True, the function will compute confidence intervals on
            the signal_rate_multiplier, for the _first_ set of extra_args.
        limit_threshold: defaults to None TODO
        inference_object_args: Passed to the initialisation of the
            InferenceObject. Defaults to {}
        refit_if_first_not_best: defaults to False
            if True, check if the first extra_args
            (assumed to be the freest extra args) is the best llhood
            and refit if not true (tolerance 1e-2)
            [not yet implemented]
        propagate_guess: If True, the previous fit results (except the values
            specified in guess) are is used as a guess, defaults to True
        return_expectations: if True, tuple of results and expectation vlues
            are returned, defaults to False
        inference_class_name: name of inference class, string
            defaults to "binference.likelihoods.ll_nt.InferenceObject"
        toydata_file: path to file containing toydata,
            defaults to None
        toydata_mode: "none", "write" or "read".
            - if "write": simulate interpolated ll and write toydata to 'toydata_file'
            - if "read": read toydata from 'toydata_file'
            - if "none": simulate interpolated ll
            Defaults to "none"
        inference_object_args: dict of arguments to be passed to the
            initialisation of the inferenceobject, defaults to {}

        neyman_threshold: name of file where the neyman threshold for
            the limit computation is stored <- NOT IMPLEMENTED

    outputs:
    --------
        array (length of extra_args array) of structured
        numpy arrays for each
    """

    #parse limit_threshod:

    if refit_if_first_not_best:
        raise NotImplementedError()

    #programatically import statistical model:
    InferenceObject = locate(inference_class_name)
    print("loaded inference class", inference_class_name,
          type(InferenceObject))
    logging.debug("Building statistical model with the following parameters:")
    logging.debug(f"wimp_mass {wimp_mass:.2f}")
    logging.debug(f"i_batch {i_batch}")
    logging.debug("livetime " + str(livetime))
    logging.debug("limit_threshold " + str(limit_threshold))
    logging.debug("toydata_file " + str(toydata_file))
    logging.debug("toydata_mode " + str(toydata_mode))
    logging.debug("signal_component_name " + str(signal_component_name))
    logging.debug("generate_args " + str(generate_args))
    logging.debug("threshold_key " + str(threshold_key))
    logging.debug("inference_object_args " + str(inference_object_args))
    
    statistical_model = InferenceObject(
        wimp_mass=wimp_mass,
        i_batch=i_batch,
        livetime=livetime,
        limit_threshold=limit_threshold,
        toydata_file=toydata_file,
        toydata_mode=toydata_mode,
        signal_component_name=signal_component_name,
        generate_args=generate_args,
        threshold_key=threshold_key,
        **inference_object_args)
    logging.debug(f"run_toymcs with expectation values per likelihood {statistical_model.get_mus()}")
    if hasattr(statistical_model, "pertoy_fix_file_generate_args"):
        logging.debug("using a per-toy fixed file, setting generate args from here")
        generate_args = statistical_model.pertoy_fix_file_generate_args
    else:
        generate_args = statistical_model.generate_args

    parameter_list = sorted(statistical_model.get_parameter_list())
    result_dtype = [(n, float) for n in parameter_list]

    try:
        additional_result_dtype = [
            (n, float) for n in statistical_model.additional_parameters
        ]
        result_dtype += additional_result_dtype
        parameter_list += statistical_model.additional_parameters
    except:
        pass

    nominal_expectations = statistical_model.get_mus()
    nominal_signal_expectation = nominal_expectations.get(
        signal_component_name, 0.)

    logging.debug("Signal expectation: " + str(signal_expectation))
    logging.debug("Nominal expectations: " + str(nominal_expectations))
    logging.debug("Nominal signal expectation: " + str(nominal_signal_expectation))
    logging.debug("generate_args: " + str(generate_args))
    if signal_expectation is not None:
        generate_args[
            signal_component_name +
            "_rate_multiplier"] = signal_expectation / nominal_signal_expectation
        logging.debug("readjusting generate_args according to signal_expectation: " + str(generate_args))

    extra_args_array = []
    if extra_args[0] == "iterate":
        parameter_name = extra_args[1]
        parameter_values = extra_args[2]
        if parameter_name.startswith("mu_"):
            parameter_name = parameter_name[3::]
            parameter_mu = nominal_expectations.get(parameter_name, 1.)
            parameter_values = np.array(parameter_values) / parameter_mu
            parameter_name = parameter_name + "_rate_multiplier"
        extra_args = [{parameter_name: pv} for pv in parameter_values]
        logging.debug(len(extra_args), extra_args)
    for extra_args_fit in extra_args:
        if extra_args_fit == "null":
            extra_args_fit = {signal_component_name + "_rate_multiplier": 0.}
        elif extra_args_fit == "true":
            extra_args_fit = {
                signal_component_name + "_rate_multiplier":
                generate_args.get(signal_component_name + "_rate_multiplier",
                                  0.)
            }
        elif extra_args_fit == "free":
            extra_args_fit = {}

        for k, i in extra_args_fit.items():
            if i == "true":
                extra_args_fit[k] = generate_args.get(k, )

        extra_args_to_arr = deepcopy(extra_args_all)
        extra_args_to_arr.update(extra_args_fit)
        extra_args_array.append(extra_args_to_arr)
    logging.debug("extra_args_array: " + str(extra_args_array))
    logging.debug("len(extra_args_array): " + str(len(extra_args_array)))

    results = [np.zeros(n_mc, dtype=result_dtype) for _ in extra_args_array]
    logging.debug("length of results list is " + str(len(results)))

    for i in tqdm(range(n_mc)):
        fit_results = statistical_model.toy_simulation(
            generate_args=generate_args,
            extra_args=extra_args_array,
            guess=guess,
            compute_confidence_interval=compute_confidence_interval,
            propagate_guess=propagate_guess)
        for fit_result, result_array in zip(fit_results, results):
            #fit_result_array = np.array([fit_result[pn] for pn in parameter_list])
            fit_result_array = tuple(fit_result[pn] for pn in parameter_list)
            result_array[i] = fit_result_array
    if toydata_mode == "write":
        statistical_model.write_toydata()
        if inference_class_name == "binference.likelihoods.ll_GOF.InferenceObject":
            statistical_model.write_reference()

    if return_expectations:
        return results, statistical_model.get_mus()
    else:
        return results


def submit_batch_toymcs():
    raise NotImplementedError()


def toymc_to_sbatch_call(
    output_filename="test_toymc.hdf5",
    n_mc=10,
    i_batch=0,
    execution_time=60,
    ram_roof=2000,  #Mb, default
    queue="xenon1t",
    singularity_container="/project2/lgrandi/xenonnt/singularity-images/xenonnt-development.simg",
    wimp_mass=50,
    signal_component_name="signal",
    signal_expectation=0.,  # no of evts.
    generate_args={},
    livetime=1.,  #ty
    extra_args=[{}],
    result_names=None,
    extra_args_all={},
    guess={},
    compute_confidence_interval=False,
    limit_threshold=None,
    threshold_key=None,
    inference_object_args={},
    refit_if_first_not_best=False,
    metadata={"version": "0.0"},
    inference_class_name="binference.likelihoods.ll_nt.InferenceObject",
    toydata_file="none",
    toydata_mode="none",
    **kwargs,
):
    """Write sbatch call for toyMC specified by the keyword arguments.
    Args:
        output_filename (str, optional): Output filename.
            Defaults to "test_toymc.hdf5".
        n_mc (int, optional): Number of toyMCs. Defaults to 10.
        i_batch (int, optional): indexes batch-dependent values.
            Defaults to 0.
        execution_time (int, optional): Limit of total run time of the job
            allocation in minutes. Defaults to 60.
        ram_roof (int, optional): Minimum memory required per allocated CPU
            in MB. Defaults to 2000.
        wimp_mass (int, optional): mass of signal in GeV. Defaults to 50.
        signal_component_name (str, optional): Name of the source.
            Defaults to "signal".
        signal_expectation (float, optional): number of events expected.
            Defaults to 0..
        livetime (float, optional): exposure in t*y. Defaults to 1..
        result_names (str, optional): TODO. Defaults to None.
        extra_args_all (dict, optional): dict of full extra_args passed
            to each fit. Defaults to {}.
        guess (dict, optional): dict passed to fitter as guess. Defaults to {}.
        compute_confidence_interval (bool, optional): if True, the function
            will compute confidence intervals on the signal_rate_multiplier,
            for the _first_ set of extra_args. Defaults to False.
        limit_threshold (float, optional): TODO. Defaults to None.
        inference_object_args (dict, optional): Dict of arguments to be
            passed to the initialisation of the inferenceobject.
            Defaults to {}.
        refit_if_first_not_best (bool, optional): If True, check if the
            first extra_args (assumed to be the freest extra args) is the
            best llhood and refit if not true (tolerance 1e-2).
            Defaults to False.
        metadata (dict, optional): TODO. Defaults to {"version":"0.0"}.
        inference_class_name (str, optional): name of inference class.
            Defaults to "binference.likelihoods.ll_nt.InferenceObject".
        toydata_file (str, optional): Path to file containing toydata.
            Defaults to "none".
        toydata_mode: "none", "write" or "read".
            - if "write": simulate interpolated ll and write toydata to 'toydata_file'
            - if "read": read toydata from 'toydata_file'
            - if "none": simulate interpolated ll
            Defaults to "none"
    Returns:
    --------
        output_filename: output filename
        call_array: sbatch call that can be submitted
    """

    log_name = splitext(output_filename)[0] + ".log"

    # ensure directory for output/logfile
    dirname = os.path.dirname(log_name)
    if not os.path.exists(dirname) and dirname != "":
        os.makedirs(dirname)

    # getting full path for sbatch script
    # this script we will write from the template-function sbatch_template()

    sbatch_name_and_path = pkg_resources.resource_filename(
        "binference", "/sbatch_submission/{filename}".format(
            filename=log_name.replace(".log", ".sbatch")))

    if not os.path.exists(os.path.dirname(sbatch_name_and_path)):
        os.makedirs(os.path.dirname(sbatch_name_and_path))

    create_sbatch_file(filename=sbatch_name_and_path,
                       queue=queue,
                       logfile=log_name,
                       execution_time=execution_time,
                       ram_roof=ram_roof,
                       singularity_container=singularity_container)

    call_array = ['sbatch', sbatch_name_and_path]

    call_array.append("--output_filename")
    call_array.append(output_filename)
    call_array.append("--n_mc")
    call_array.append("{:d}".format(n_mc))
    call_array.append("--i_batch")
    call_array.append("{:d}".format(i_batch))
    call_array.append("--wimp_mass")
    call_array.append("{:d}".format(wimp_mass))
    call_array.append("--signal_expectation")
    if signal_expectation is None:
        call_array.append("None")
    else:
        call_array.append("{:.2f}".format(signal_expectation))
    call_array.append("--signal_component_name")
    call_array.append("{:s}".format(signal_component_name))
    call_array.append("--result_names")
    if result_names is None:
        call_array.append("None")
    else:
        call_array.append(dumps(result_names).replace(" ", ""))
    call_array.append("--generate_args")
    call_array.append(dumps(generate_args).replace(" ", ""))
    call_array.append("--livetime")
    if livetime is None:
        call_array.append("None")
    else:
        call_array.append(dumps(livetime).replace(" ", ""))
    call_array.append("--extra_args")
    call_array.append(dumps(extra_args).replace(" ", ""))
    call_array.append("--extra_args_all")
    call_array.append(dumps(extra_args_all).replace(" ", ""))
    call_array.append("--guess")
    call_array.append(dumps(guess).replace(" ", ""))
    if compute_confidence_interval:
        call_array.append("--compute_confidence_interval")
        call_array.append("1")
    call_array.append("--limit_threshold")
    if limit_threshold is None:
        call_array.append("None")
    else:
        call_array.append(limit_threshold)
    call_array.append("--inference_object_args")
    call_array.append(dumps(inference_object_args).replace(" ", ""))
    if refit_if_first_not_best:
        call_array.append("--refit_if_first_not_best")
        call_array.append("1")
    call_array.append("--metadata")
    call_array.append(dumps(metadata).replace(" ", ""))
    call_array.append("--inference_class_name")
    call_array.append(inference_class_name)
    call_array.append("--toydata_file")
    call_array.append(toydata_file)
    call_array.append("--toydata_mode")
    call_array.append(toydata_mode)
    call_array.append("--threshold_key")
    if threshold_key is None:
        call_array.append("None")
    else:
        call_array.append(threshold_key)

    return output_filename, call_array


def toymc_to_sbatch_call_array_update(
        parameters_to_vary={},
        parameters_to_zip={},
        parameters_in_common={},
        wildcards_for_threshold=["signal_rate_multiplier", "signal_expectation", "n_mc", "n_batch"]):

    output_filename = parameters_in_common.get("output_filename")
    file_name_pattern_threshold = get_filename_pattern_threshold(
        output_filename=output_filename, wildcards_for_threshold=wildcards_for_threshold)

    merged_combinations = binference.utils.compute_variations(
        parameters_in_common=parameters_in_common,
        parameters_to_vary=parameters_to_vary,
        parameters_to_zip=parameters_to_zip)

    #find run toyMC default args:
    callargs, _, _, calldefaults = inspect.getargspec(toymc_to_sbatch_call)
    # ignore the deprecation warning for now, update can be done below
    #  signature = inspect.signature(toymc_to_sbatch_call)
    default_args = dict(zip(callargs, calldefaults))
    default_args["n_batch"] = 1
    default_args["output_filename"] = default_args.get("output_filename",
                                                       "test_toymc.hdf5")
    default_args["n_mc"] = default_args.get("n_mc", 10)

    fnames, call_arrays, function_args_to_return = [], [], []
    for combination in tqdm(merged_combinations):
        function_args = deepcopy(default_args)
        mergedeep.merge(function_args, combination)  # update defaults with combination
        function_args = binference.utils.flatten_function_args(combination=combination, function_args=function_args)

        function_args["n_mc"] = int(function_args["n_mc"] /
                                    function_args["n_batch"])

        filename_pattern = function_args["output_filename"]
        filename_pattern = add_imc(
            filename_pattern)  #so that we'll index by batch
        n_batch = function_args["n_batch"]
        function_args.pop("n_batch", None)
        toydata_file_pattern = function_args.get("toydata_file", "none")
        toydata_file_pattern = add_imc(
            toydata_file_pattern)  #so that we'll index by batch

        # do the same for filenames passed in inference_object_args:
        toy_reference_file_pattern = None
        hist_reference_file_pattern = None
        hist_data_file_pattern = None
        filenames = {
            'toy_reference_file': toy_reference_file_pattern,
            'hist_reference_file': hist_reference_file_pattern,
            'hist_data_file': hist_data_file_pattern
        }

        for filename_key in filenames.keys():
            for key in function_args["inference_object_args"].keys():
                if key == filename_key:
                    filenames[filename_key] = (
                        function_args["inference_object_args"][key])
                    filenames[filename_key] = add_imc(filenames[filename_key])

        #filename = filename.format(**function_args)
        #toydata_file = toydata_file.format(**function_args)
        for i_batch in range(n_batch):
            function_args["i_batch"] = i_batch

            function_args["toydata_file"] = toydata_file_pattern.format(
                **function_args)
            function_args["output_filename"] = filename_pattern.format(
                **function_args)
            for filename_key, file_pattern in filenames.items():
                if file_pattern is not None:
                    file = file_pattern.format(**function_args)
                    function_args["inference_object_args"][filename_key] = file
            threshold_key, _, _ = generate_threshold_key(
                file_name_pattern=file_name_pattern_threshold,
                function_args=function_args)
            function_args["threshold_key"] = threshold_key
            fname, call_array = toymc_to_sbatch_call(**function_args)
            fnames.append(fname)
            call_arrays.append(call_array)
    return fnames, call_arrays


def toymc_to_sbatch_call_array(parameters_to_vary={},
                               parameters_to_zip={},
                               parameters_in_common={}):
    """
    Wrapper for `toymc_to_sbatch_call`.
    Function that runs `toymc_to_sbatch_call` for the product of all
    iterables in parameters_to_vary, keeping parameters_in_common fixed.
    parameters_to_zip are passed to zip-- so varied together.
    Possible keys of the dicts are the keyword arguments of
    `toymc_to_sbatch_call`.

    input args:
    -----------
    parameters_to_vary: dict of parameters that are varied. An sbatch call
        is generated for all possible combinations pf parameters_to_vary,
        defaults to {}
    parameters_to_zip: dict of parameters that are varied together,
        defaults to {}
    parameters_in_common: dict of parameters that are fixed, defaults to {}

    outputs:
    --------
    fnames: array of output filenemes
    call_arrays: array of sbatch calls that can be submitted
    """

    input_parameters_to_vary = copy.deepcopy(parameters_to_vary)

    #find run toyMC default args:
    callargs, _, _, calldefaults = inspect.getargspec(toymc_to_sbatch_call)
    default_args = dict(zip(callargs, calldefaults))
    default_args["n_batch"] = 1
    default_args.update(parameters_in_common)
    default_args["output_filename"] = default_args.get("output_filename",
                                                       "test_toymc.hdf5")
    default_args["n_mc"] = default_args.get("n_mc", 10)
    call_arrays = []
    fnames = []

    parameters_to_vary_names = sorted(parameters_to_vary.keys())
    for k in parameters_to_vary_names:
        if isinstance(parameters_to_vary[k], dict):
            # allows variations inside of dicts
            parameters_to_vary[k] = [
                item for item in binference.utils.dict_product(
                    parameters_to_vary[k])
            ]
        else:
            parameters_to_vary[k] = parameters_to_vary[k]

    parameters_to_vary_values = [
        parameters_to_vary[k] for k in parameters_to_vary_names
    ]
    parameters_to_zip_names = sorted(parameters_to_zip.keys())
    parameters_to_zip_values = [
        parameters_to_zip[k] for k in parameters_to_zip_names
    ]

    # loop through all possible combinations of parameters_to_vary_values
    #
    if len(parameters_to_vary_names) == 0:
        iter_product = [0]
    else:
        iter_product = product(*parameters_to_vary_values)
    for parameters_to_vary_value in iter_product:
        if len(parameters_to_zip_names) == 0:
            iter_zip = [0]
        else:
            iter_zip = zip(*parameters_to_zip_values)

        for parameters_to_zip_value in iter_zip:

            function_args = deepcopy(default_args)
            # overwrite default arguments in function_args with
            # parameters_to_vary and parameters_to_zip:
            if 0 < len(parameters_to_vary_names):
                parameters_to_vary_dict = {
                    pn: pv
                    for pn, pv in zip(parameters_to_vary_names,
                                      parameters_to_vary_value)
                }

                #function to allow to set dict arguments from multiple places:
                for pn in set(function_args.keys()) & set(
                        parameters_to_vary_dict.keys()):
                    if (type(function_args[pn]) == dict) and (type(
                            parameters_to_vary_dict[pn]) == dict):
                        parameters_to_vary_dict[pn] = dict(
                            function_args[pn], **parameters_to_vary_dict[pn])

                        # add all keys of dicts as function_args
                        # needed to allow using parameters from e.g.
                        # generate_args in the output_filename
                        for key, value in parameters_to_vary_dict[pn].items():
                            function_args[key] = value

                function_args.update(parameters_to_vary_dict)

            if 0 < len(parameters_to_zip_names):
                parameters_to_zip_dict = {
                    pn: pv
                    for pn, pv in zip(parameters_to_zip_names,
                                      parameters_to_zip_value)
                }
                for pn in set(function_args.keys()) & set(
                        parameters_to_zip_dict.keys()):
                    if (type(function_args[pn]) == dict) and (type(
                            parameters_to_zip_dict[pn]) == dict):
                        parameters_to_zip_dict[pn] = dict(
                            function_args[pn], **parameters_to_zip_dict[pn])

                function_args.update(parameters_to_zip_dict)
                #  for key in parameters_to_zip_dict.keys():
                #      if key not in function_args.keys():
                #          function_args["generate_args"][key] = parameters_to_zip_dict[key]
                #  function_args["generate_args"].update(parameters_to_zip_dict)

            #for pn in set(function_args.keys(), parameters_to_vary_names):
            #    if type(function_args)Opv

            #if 0 < len(parameters_to_vary_names):
            #    function_args.update(parameters_to_vary_dict)
            #if 0 < len(parameters_to_zip_names):
            #    function_args.update({pn: pv for pn, pv in zip(parameters_to_zip_names, parameters_to_zip_value)})

            function_args["n_mc"] = int(function_args["n_mc"] /
                                        function_args["n_batch"])
            filename_pattern = function_args["output_filename"]
            filename_pattern = add_imc(
                filename_pattern)  #so that we'll index by batch
            n_batch = function_args["n_batch"]
            function_args.pop("n_batch", None)
            toydata_file_pattern = function_args.get("toydata_file", "none")
            toydata_file_pattern = add_imc(
                toydata_file_pattern)  #so that we'll index by batch
            # do the same for filenames passed in inference_object_args:
            toy_reference_file_pattern = None
            hist_reference_file_pattern = None
            hist_data_file_pattern = None
            filenames = {
                'toy_reference_file': toy_reference_file_pattern,
                'hist_reference_file': hist_reference_file_pattern,
                'hist_data_file': hist_data_file_pattern
            }

            for filename_key in filenames.keys():
                for key in function_args["inference_object_args"].keys():
                    if key == filename_key:
                        filenames[filename_key] = (
                            function_args["inference_object_args"][key])
                        filenames[filename_key] = add_imc(
                            filenames[filename_key])

            #filename = filename.format(**function_args)
            #toydata_file = toydata_file.format(**function_args)
            for i_batch in range(n_batch):
                function_args["i_batch"] = i_batch

                function_args["toydata_file"] = toydata_file_pattern.format(
                    **function_args)
                function_args["output_filename"] = filename_pattern.format(
                    **function_args)
                for filename_key, file_pattern in filenames.items():
                    if file_pattern is not None:
                        file = file_pattern.format(**function_args)
                        function_args["inference_object_args"][
                            filename_key] = file

                fname, call_array = toymc_to_sbatch_call(**function_args)
                fnames.append(fname)
                call_arrays.append(call_array)
    return fnames, call_arrays


def parse_cl_arguments(arguments):
    parser = argparse.ArgumentParser(
        description="command line running of run_toymcs")

    parser.add_argument('--output_filename',
                        type=str,
                        required=True,
                        help="path where the output is stored")
    parser.add_argument('--result_names', default="None")
    parser.add_argument('--n_mc',
                        type=int,
                        default=10,
                        help="number of MC simulations")
    parser.add_argument('--i_batch',
                        type=int,
                        default=0,
                        help="number of batches")
    parser.add_argument('--wimp_mass',
                        type=int,
                        default=50,
                        help="WIMP mass in GeV")
    parser.add_argument("--signal_expectation", type=str, default="None")
    parser.add_argument("--signal_component_name", type=str, default="signal")
    parser.add_argument('--generate_args',
                        type=str,
                        default="{}",
                        help="arguments for the toy generation")
    parser.add_argument(
        '--livetime',
        type=str,
        default='None',  # uses definiton from the likelihood
        help=
        "livetime in tonne x year, if multiple livetimes are given (as a list) then the livetime per sub-likelihood will be set accoringly"
    )  #ty
    parser.add_argument('--extra_args',
                        type=str,
                        default='["null","free"]',
                        help="fitting arguments")
    parser.add_argument('--extra_args_all',
                        type=str,
                        default='{}',
                        help="fitting arguments shared")
    parser.add_argument('--guess',
                        type=str,
                        default='{}',
                        help='initial guess')
    parser.add_argument(
        '--compute_confidence_interval', type=bool,
        default=False)  #WARNING: --compute_ul False will s--compute_ul to True
    parser.add_argument('--limit_threshold', type=str, default='None')
    parser.add_argument('--inference_object_args', type=str, default='{}')
    parser.add_argument(
        '--refit_if_first_not_best', type=bool,
        default=False)  #WARNING: --compute_ul False will s--compute_ul to True
    parser.add_argument('--metadata', type=str, default='{"version":"0.0"}')

    parser.add_argument('--inference_class_name', type=str, default="none")
    parser.add_argument('--toydata_file', type=str, default="none")
    parser.add_argument('--toydata_mode', type=str, default="none")
    parser.add_argument('--threshold_key', type=str, default="none")

    args = parser.parse_args(arguments)

    args.output_filename = args.output_filename
    if args.result_names == "None":
        args.result_names = "None"
    else:
        args.result_names = loads(args.result_names)
    args.n_mc = args.n_mc
    args.i_batch = args.i_batch
    args.wimp_mass = args.wimp_mass
    args.signal_expectation = args.signal_expectation
    args.signal_expectation = None if args.signal_expectation == "None" else float(
        args.signal_expectation)
    args.signal_component_name = args.signal_component_name
    args.generate_args = loads(args.generate_args)
    if args.livetime == 'None':
        args.livetime = None
    else:
        args.livetime = loads(args.livetime)
    args.extra_args = loads(args.extra_args)
    args.extra_args_all = loads(args.extra_args_all)
    args.guess = loads(args.guess)
    args.inference_object_args = loads(args.inference_object_args)
    args.compute_confidence_interval = args.compute_confidence_interval
    args.limit_threshold = args.limit_threshold
    args.limit_threshold = None if args.limit_threshold == "None" else args.limit_threshold
    args.refit_if_first_not_best = args.refit_if_first_not_best
    args.metadata = loads(args.metadata)
    args.inference_class_name = args.inference_class_name
    args.toydata_file = args.toydata_file
    args.toydata_mode = args.toydata_mode

    return args


def run_toymcs_from_cl(arguments):
    """Run `run_toymcs` with inputs provided in the command line and wirte the
    results in an output file.
    """
    args = parse_cl_arguments(arguments=arguments)
    result_names = args.result_names
    logging.debug("args are")
    logging.debug(args)

    results, nominal_expectations = run_toymcs(
        n_mc=args.n_mc,
        i_batch=args.i_batch,
        wimp_mass=args.wimp_mass,
        signal_expectation=args.signal_expectation,  # no of evts.
        signal_component_name=args.signal_component_name,
        generate_args=args.generate_args,
        livetime=args.livetime,  #ty
        extra_args=args.extra_args,
        extra_args_all=args.extra_args_all,
        guess=args.guess,
        compute_confidence_interval=args.compute_confidence_interval,
        limit_threshold=args.limit_threshold,
        inference_object_args=args.inference_object_args,
        refit_if_first_not_best=args.refit_if_first_not_best,
        return_expectations=True,
        inference_class_name=args.inference_class_name,
        toydata_mode=args.toydata_mode,
        toydata_file=args.toydata_file,
        threshold_key=args.threshold_key)

    if (args.extra_args[0] == "iterate") and (result_names == "None"):
        logging.debug("extraargs is", args.extra_args)
        result_names = ["{:.3f}".format(float(v)) for v in args.extra_args[2]]
        logging.debug("result_names", result_names)
    if result_names == "None":
        result_names = ["{:d}".format(i) for i in range(len(args.extra_args))]
        for i, ea in enumerate(
                args.extra_args
        ):  #if using named extra args (free, null, true), use that name
            if ea in ["null", "free", "true"]:
                result_names[i] = ea
    args.metadata["date"] = datetime.now().strftime('%Y%m%d_%H:%M:%S')
    args.metadata["generate_args"] = args.generate_args
    args.metadata["signal_expectation"] = args.signal_expectation
    args.metadata["signal_component_name"] = args.signal_component_name
    args.metadata["nominal_expectations"] = nominal_expectations
    if args.extra_args[0] == "iterate":
        eas = [{args.extra_args[1]: v} for v in args.extra_args[2]]
        array_metadatas = [{"extra_args": ea} for ea in eas]
    else:
        array_metadatas = [{"extra_args": ea} for ea in args.extra_args]
    logging.debug(f"End of run_toymcs_from_cl: len(result_names),len(results) = ({len(result_names)}, {len(results)})")
    numpy_arrays_and_names = [(r, rn) for rn, r in zip(result_names, results)]
    logging.debug("Results stored to file:")
    logging.debug("signal_expectation: " + str(args.signal_expectation))
    logging.debug("nominal expectations: " + str(nominal_expectations))
    logging.debug("generate_args: " + str(args.generate_args))
    logging.debug("numpy_arrays_and_names: " + str(numpy_arrays_and_names))
    
    for r, rn in zip(results, result_names):
        logging.debug(str(rn) + str(r))
    # logging.debug(numpy_arrays_and_names)
    # logging.debug(len(numpy_arrays_and_names))
    logging.debug("Metadata: " + str(array_metadatas))

    print(f'Saving {args.output_filename}')
    numpy_to_toyfile(args.output_filename,
                     numpy_arrays_and_names=numpy_arrays_and_names,
                     metadata=args.metadata,
                     array_metadatas=array_metadatas)


def compute_neyman_thresholds(
    file_name_pattern,
    threshold_name="thresholds.hdf5",
    parameters_to_vary={},
    parameters_to_zip={},
    parameters_in_common={},
    parameters_as_wildcards=["signal_rate_multiplier", "signal_expectation", "n_mc", "n_batch"],
    signal_component_name="signal",
    confidence_levels=[0.8, 0.9, 0.95],
    free_name="free",
    null_name="true",
    one_sided=False,
    one_sided_minimum_mu=1.,
    return_to_dict=False,
    metadata={
        "version": "0.0",
        "date": datetime.now().strftime('%Y%m%d_%H:%M:%S')
    }):
    """
        function to run over any number of toyMC results, computing the llr between free_name and null_name and
        selecting all with same parameters ordering by signal expectation
        Result is stored, labeled by the parameters in "parameters_to_vary"
        if one_sided: llrs with sbest<strue are set to 0, and below one_sided_minimum_mu, the threshold is set to the
        max within that range.
    """

    return_dict = {}
    #put wildcards for every key in searchkeys:
    #.*? = "any character (.)", "repeated (*)", in a non-greedy way (?)
    for sk in parameters_as_wildcards:
        file_name_pattern = re.sub("\{" + sk + ".*?\}", "*", file_name_pattern)

    #use the labels of parameters_to_vary as labels for neyman thresholds:
    threshold_key_names = re.findall("\{(.*?)\}", file_name_pattern)
    threshold_key_names = sorted(
        [item.split(":")[0] for item in threshold_key_names])
    #  threshold_key_names = sorted(list(parameters_to_vary.keys()))
    threshold_key_pattern = "_".join(
        [tkn + "_{" + tkn + ":s}" for tkn in threshold_key_names])
    if len(threshold_key_names) == 0:
        threshold_key_pattern = "thresholds"

    #find run toyMC default args:
    callargs, _, _, calldefaults = inspect.getargspec(toymc_to_sbatch_call)
    default_args = dict(zip(callargs, calldefaults))
    default_args["n_batch"] = 1
    default_args.update(parameters_in_common)
    default_args["n_mc"] = default_args.get("n_mc", 10)
    parameters_to_vary_names = sorted(parameters_to_vary.keys())
    for k in parameters_to_vary_names:
        if isinstance(parameters_to_vary[k], dict):
            # allows variations inside of dicts
            parameters_to_vary[k] = [
                item for item in binference.utils.dict_product(
                    parameters_to_vary[k])
            ]
        else:
            parameters_to_vary[k] = parameters_to_vary[k]
    parameters_to_vary_values = [
        parameters_to_vary[k] for k in parameters_to_vary_names
    ]
    parameters_to_zip_names = sorted(parameters_to_zip.keys())
    parameters_to_zip_values = [
        parameters_to_zip[k] for k in parameters_to_zip_names
    ]

    if len(parameters_to_vary_names) == 0:
        iter_product = [0]
    else:
        iter_product = product(*parameters_to_vary_values)

    for parameters_to_vary_value in iter_product:
        function_args = deepcopy(default_args)
        for pn, pv in zip(parameters_to_vary_names, parameters_to_vary_value):
            function_args.update({pn: pv})
            function_args[pn] = pv
        # overwrite default arguments in function_args with
        # parameters_to_vary and parameters_to_zip:
        if 0 < len(parameters_to_vary_names):
            parameters_to_vary_dict = {
                pn: pv
                for pn, pv in zip(parameters_to_vary_names,
                                  parameters_to_vary_value)
            }

            #function to allow to set dict arguments from multiple places:
            for pn in set(function_args.keys()) & set(
                    parameters_to_vary_dict.keys()):
                if (type(function_args[pn]) == dict) and (type(
                        parameters_to_vary_dict[pn]) == dict):
                    parameters_to_vary_dict[pn] = dict(
                        function_args[pn], **parameters_to_vary_dict[pn])

                    # add all keys of dicts as function_args
                    # needed to allow using parameters from e.g.
                    # generate_args in the output_filename
                    for key, value in parameters_to_vary_dict[pn].items():
                        function_args[key] = value

            function_args.update(parameters_to_vary_dict)

        if len(parameters_to_zip_names) == 0:
            iter_zip = [0]
        else:
            iter_zip = zip(*parameters_to_zip_values)

        for parameters_to_zip_value in iter_zip:
            if 0 < len(parameters_to_zip_names):
                parameters_to_zip_dict = {
                    pn: pv
                    for pn, pv in zip(parameters_to_zip_names,
                                      parameters_to_zip_value)
                }
                for pn in set(function_args.keys()) & set(
                        parameters_to_zip_dict.keys()):
                    if (type(function_args[pn]) == dict) and (type(
                            parameters_to_zip_dict[pn]) == dict):
                        parameters_to_zip_dict[pn] = dict(
                            function_args[pn], **parameters_to_zip_dict[pn])

                function_args.update(parameters_to_zip_dict)

            llrs_dict = dict()

            file_pattern = file_name_pattern.format(**function_args)
            print("Reading files for pattern:", file_pattern)

            threshold_key_values = {
                tkn: dumps(function_args[tkn])
                for tkn in threshold_key_names
            }
            threshold_key = threshold_key_pattern.format(
                **threshold_key_values)

            file_list = glob(file_pattern)
            nominal_signal_expectations = []
            if True:
                for file in file_list:
                    with h5py.File(file, "r") as f:
                        signal_expectation = float(
                            f.attrs["signal_expectation"])
                        nominal_signal_expectation = loads(
                            f.attrs["nominal_expectations"]
                        )[signal_component_name]
                        nominal_signal_expectations.append(
                            nominal_signal_expectation)
                        llfree = f["fits/" + free_name][()]["ll"]
                        llnull = f["fits/" + null_name][()]["ll"]
                        llr = 2. * (llfree - llnull)

                        if one_sided:
                            signal_rate_multiplier = signal_expectation / nominal_signal_expectation
                            shats = f["fits/" +
                                      free_name][()]["signal_rate_multiplier"]
                            llr[shats < signal_rate_multiplier] = 0.

                        llrs_dict[signal_expectation] = [llr] + llrs_dict.get(
                            signal_expectation, [])

                nominal_signal_expectations = np.array(
                    nominal_signal_expectations)
                assert nominal_signal_expectations.std(
                ) / nominal_signal_expectations.mean() < 1e-6
                signal_expectations = np.array(sorted(llrs_dict.keys()))
                for k, i in llrs_dict.items():
                    llrs_dict[k] = np.concatenate(i)

                thresholds_dict = {
                    cl: np.zeros(len(signal_expectations))
                    for cl in confidence_levels
                }
                for i, signal_expectation in enumerate(signal_expectations):
                    llrs = llrs_dict[signal_expectation]
                    for confidence_level in confidence_levels:
                        thresholds_dict[confidence_level][i] = np.percentile(
                            llrs, 100. * confidence_level)

                #Apply a very-smallest-mu-threshold (stand-in for PCL downstream, and eases minimizer trouble for ULs)
                if one_sided:
                    for confidence_level in confidence_levels:
                        threshold = thresholds_dict[confidence_level]
                        threshold[
                            signal_expectations <
                            one_sided_minimum_mu] = threshold[
                                one_sided_minimum_mu < signal_expectations][0]
                        thresholds_dict[confidence_level] = threshold

            return_dict[
                threshold_key] = thresholds_dict, signal_expectations, nominal_signal_expectations.mean(
                )

            #except:
            #    pass
    if return_to_dict:
        return return_dict
    else:
        #store this stuff in a new hdf5!
        with h5py.File(threshold_name, "w") as f:
            for k, i in metadata.items():
                f.attrs[k] = i
            f.attrs["threshold_key_names"] = dumps(threshold_key_names)
            for threshold_name, (
                    thresholds_dict, signal_expectations,
                    nominal_signal_expectation) in return_dict.items():
                dset = f.create_dataset(threshold_name +
                                        "/signal_expectations",
                                        data=signal_expectations)
                dset.attrs["nominal_signal_expectation"] = dumps(
                    nominal_signal_expectation)
                for confidence_level, threshold in thresholds_dict.items():
                    dset = f.create_dataset(
                        threshold_name +
                        "/threshold_cl_{:.2f}".format(confidence_level),
                        data=threshold)
                    dset.attrs["confidence_level"] = dumps(confidence_level)


def compute_neyman_thresholds_update(
    file_name_pattern,
    threshold_name="thresholds.hdf5",
    parameters_to_vary={},
    parameters_to_zip={},
    parameters_in_common={},
    parameters_as_wildcards=["signal_rate_multiplier", "signal_expectation", "n_mc", "n_batch"],
    signal_component_name="signal",
    confidence_levels=[0.8, 0.9, 0.95],
    free_name="free",
    null_name="true",
    one_sided=False,
    one_sided_minimum_mu=1.,
    return_to_dict=False,
    metadata={
        "version": "0.0",
        "date": datetime.now().strftime('%Y%m%d_%H:%M:%S')
    }):
    """
        function to run over any number of toyMC results, computing the llr between free_name and null_name and
        selecting all with same parameters ordering by signal expectation
        Result is stored, labeled by the parameters in "parameters_to_vary"
        if one_sided: llrs with sbest<strue are set to 0, and below one_sided_minimum_mu, the threshold is set to the
        max within that range.
    """

    return_dict = {}
    #put wildcards for every key in searchkeys:
    #.*? = "any character (.)", "repeated (*)", in a non-greedy way (?)
    for sk in parameters_as_wildcards:
        file_name_pattern = re.sub("\{" + sk + ".*?\}", "*", file_name_pattern)

    merged_combinations = binference.utils.compute_variations(
        parameters_in_common=parameters_in_common,
        parameters_to_vary=parameters_to_vary,
        parameters_to_zip=parameters_to_zip,
    )

    #find run toyMC default args:
    callargs, _, _, calldefaults = inspect.getargspec(toymc_to_sbatch_call)
    default_args = dict(zip(callargs, calldefaults))
    default_args["n_batch"] = 1
    default_args.update(parameters_in_common)
    default_args["n_mc"] = default_args.get("n_mc", 10)

    for combination in merged_combinations:

        function_args = copy.deepcopy(default_args)
        mergedeep.merge(function_args, combination)
        function_args = binference.utils.flatten_function_args(combination, function_args)

        threshold_key, threshold_key_pattern, threshold_key_names = generate_threshold_key(
            file_name_pattern=file_name_pattern, function_args=function_args)
        file_pattern = file_name_pattern.format(**function_args)
        print("Reading files for pattern:", file_pattern)

        llrs_dict = dict()

        file_list = glob(file_pattern)
        nominal_signal_expectations = []
        for file in file_list:
            with h5py.File(file, "r") as f:
                nominal_signal_expectation = loads(
                    f.attrs["nominal_expectations"])[signal_component_name]
                try:
                    signal_expectation = float(f.attrs["signal_expectation"])
                except:
                    generate_args = loads(f.attrs["generate_args"])
                    signal_rate_multiplier = generate_args[signal_component_name + "_rate_multiplier"]
                    signal_expectation = nominal_signal_expectation * signal_rate_multiplier
                nominal_signal_expectations.append(nominal_signal_expectation)
                llfree = f["fits/" + free_name][()]["ll"]
                llnull = f["fits/" + null_name][()]["ll"]
                llr = 2. * (llfree - llnull)

                if one_sided:
                    signal_rate_multiplier = signal_expectation / nominal_signal_expectation
                    shats = f["fits/" +
                              free_name][()]["signal_rate_multiplier"]
                    llr[shats < signal_rate_multiplier] = 0.

                llrs_dict[signal_expectation] = [llr] + llrs_dict.get(
                    signal_expectation, [])

        nominal_signal_expectations = np.array(nominal_signal_expectations)
        assert nominal_signal_expectations.std(
        ) / nominal_signal_expectations.mean() < 1e-6
        signal_expectations = np.array(sorted(llrs_dict.keys()))
        for k, i in llrs_dict.items():
            llrs_dict[k] = np.concatenate(i)

        thresholds_dict = {
            cl: np.zeros(len(signal_expectations))
            for cl in confidence_levels
        }
        for i, signal_expectation in enumerate(signal_expectations):
            llrs = llrs_dict[signal_expectation]
            for confidence_level in confidence_levels:
                thresholds_dict[confidence_level][i] = np.percentile(
                    llrs, 100. * confidence_level)

        #Apply a very-smallest-mu-threshold (stand-in for PCL downstream, and eases minimizer trouble for ULs)
        if one_sided:
            for confidence_level in confidence_levels:
                threshold = thresholds_dict[confidence_level]
                threshold[
                    signal_expectations < one_sided_minimum_mu] = threshold[
                        one_sided_minimum_mu < signal_expectations][0]
                thresholds_dict[confidence_level] = threshold

        return_dict[
            threshold_key] = thresholds_dict, signal_expectations, nominal_signal_expectations.mean(
            )

    if return_to_dict:
        return return_dict
    else:
        #store this stuff in a new hdf5!
        with h5py.File(threshold_name, "w") as f:
            for k, i in metadata.items():
                f.attrs[k] = i
            f.attrs["threshold_key_names"] = dumps(threshold_key_names)
            f.attrs["threshold_key_pattern"] = dumps(threshold_key_pattern)
            for threshold_name, (
                    thresholds_dict, signal_expectations,
                    nominal_signal_expectation) in return_dict.items():
                dset = f.create_dataset(threshold_name +
                                        "/signal_expectations",
                                        data=signal_expectations)
                dset.attrs["nominal_signal_expectation"] = dumps(
                    nominal_signal_expectation)
                for confidence_level, threshold in thresholds_dict.items():
                    dset = f.create_dataset(
                        threshold_name +
                        "/threshold_cl_{:.2f}".format(confidence_level),
                        data=threshold)
                    dset.attrs["confidence_level"] = dumps(confidence_level)


def generate_threshold_key(file_name_pattern, function_args):
    threshold_key_pattern, threshold_key_names = generate_threshold_key_pattern(file_name_pattern=file_name_pattern)
    threshold_key = threshold_key_pattern.format(**function_args)

    return threshold_key, threshold_key_pattern, threshold_key_names

def get_filename_pattern_threshold(output_filename,
        wildcards_for_threshold=["signal_rate_multiplier", "signal_expectation", "n_mc", "n_batch"]):
    file_name_pattern_threshold = output_filename.split(
        ".hdf5")[0] + "_{n_batch:d}.hdf5"
    for sk in wildcards_for_threshold:
        file_name_pattern_threshold = re.sub("\{" + sk + ".*?\}", "*",
                                             file_name_pattern_threshold)
    return file_name_pattern_threshold

def generate_threshold_key_pattern(file_name_pattern):
    threshold_key_variable = re.findall("\{(.*?)\}", file_name_pattern)
    threshold_key_names = [
        item.split(":")[0] for item in threshold_key_variable
    ]
    zipped_thresholds = sorted(zip(threshold_key_names,
                                   threshold_key_variable))

    threshold_key_pattern = "_".join(
        [tkn + "_{" + var + "}" for tkn, var in zipped_thresholds])
    return threshold_key_pattern, threshold_key_names
