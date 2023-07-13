from inference_interface import toydata_from_file, structured_array_to_dict
import h5py
from json import loads, dumps
from tqdm import tqdm
from subprocess import call, check_output
from os import environ
import warnings
from time import sleep
from pathlib import Path
from random import shuffle
import yaml
import pickle
import socket
import os
from pydoc import locate
import numpy as np
import copy
import itertools
import mergedeep
import tqdm
import re
import pandas as pd
import scipy.stats as sps
import logging
logging.basicConfig(level=logging.DEBUG)


def read_pandas_data(iname, transforms, format="pickle"):
    """
        function to read .hdf file containing pandas dataframe of events,
        return pandas df with inclusion of all the newly transformed values.
        Example: giving transform {'z':lambda pdata: pdata.z_3d_nn} will copy the z_3d_nn value into a column named z
    """
    if format == "pickle":
        pdata = pd.read_pickle(iname)
    elif format == "hdf":
        pdata = pd.read_hdf(iname, 'table')
    else:
        raise SystemExit("format not supported")
    vs = {k: f(pdata) for k, f in transforms.items()}
    pdata = pdata.assign(**vs)
    return pdata


def read_neyman_threshold_update(file_name,
                                 threshold_key,
                                 confidence_level=0.9):
    """
        Function to load Neyman threshold from function
    """
    with h5py.File(file_name, "r") as f:
        if threshold_key is not None:
            #  threshold_key_names = f.attrs["threshold_key_names"]
            #  threshold_key_pattern = f.attrs["threshold_key_pattern"]
            #
            #  threshold_key = threshold_key_pattern.format(
            #      **threshold_key_values)
            try:
                signal_expectations = f[threshold_key +
                                        "/signal_expectations"][()]
            except KeyError:
                print("key:", threshold_key, "not found.")
                print("Available keys are:", )
                for key in f.keys():
                    print(key)
                raise SystemExit
            nominal_signal_expectation = \
                loads(f[threshold_key + "/signal_expectations"].attrs["nominal_signal_expectation"])
            thresholds = f[threshold_key +
                           "/threshold_cl_{:.2f}".format(confidence_level)][()]
            return signal_expectations, thresholds, nominal_signal_expectation
        else:
            ret = dict()
            for threshold_key in f.keys():
                signal_expectations = f[threshold_key +
                                        "/signal_expectations"][()]
                nominal_signal_expectation = \
                    loads(f[threshold_key + "/signal_expectations"].attrs["nominal_signal_expectation"])
                thresholds = f[
                    threshold_key +
                    "/threshold_cl_{:.2f}".format(confidence_level)][()]
                ret[threshold_key] = signal_expectations, thresholds, nominal_signal_expectation
            return ret


def read_neyman_threshold(file_name,
                          parameter_values=None,
                          confidence_level=0.9):
    """
        Function to load Neyman threshold from function
    """
    with h5py.File(file_name, "r") as f:

        if parameter_values is not None:
            threshold_key_names = loads(f.attrs["threshold_key_names"])
            threshold_key_pattern = "_".join(
                [tkn + "_{" + tkn + ":s}" for tkn in threshold_key_names])
            threshold_key_values = {
                tkn: dumps(parameter_values[tkn])
                for tkn in threshold_key_names
            }
            threshold_key = threshold_key_pattern.format(
                **threshold_key_values)
            try:
                signal_expectations = f[threshold_key +
                                        "/signal_expectations"][()]
            except KeyError:
                print("key:", threshold_key, "not found.")
                print("Available keys are:", )
                for key in f.keys():
                    print(key)
                raise SystemExit
            nominal_signal_expectation = \
                loads(f[threshold_key + "/signal_expectations"].attrs["nominal_signal_expectation"])
            thresholds = f[threshold_key +
                           "/threshold_cl_{:.2f}".format(confidence_level)][()]
            return signal_expectations, thresholds, nominal_signal_expectation
        else:
            ret = dict()
            for threshold_key in f.keys():
                signal_expectations = f[threshold_key +
                                        "/signal_expectations"][()]
                nominal_signal_expectation = \
                    loads(f[threshold_key + "/signal_expectations"].attrs["nominal_signal_expectation"])
                thresholds = f[
                    threshold_key +
                    "/threshold_cl_{:.2f}".format(confidence_level)][()]
                ret[threshold_key] = signal_expectations, thresholds, nominal_signal_expectation
            return ret


def number_of_sbatch_jobs():
    try:
        uname = environ.get('USER')
        a = check_output(
            ["/software/slurm-current-el7-x86_64/bin/squeue", "-u", uname])
        return len(str(a).split("\\n")) - 1
    except:
        return np.inf


def submit_commandline_calls(
    call_arrays=[],
    filename_array=None,  #if not None, will check if each fname exists and only call if not
    max_jobs=200,  # number of max sbatch calls
    wait_time=0.01,  #s between each submission call
    wait_if_full=0.2 * 60,  #time to wait between checking sbatch queue
    scramble_run_order=True,  # it is sometimes convenient to submit in random order to optimise that PDFs are
    # cached and to gradually build up the entire result
    run_if_file_exists=False):
    """
        Function to take a list of calls (and optionally a list of filenames to submit only if the file is not present)
        and call each in turn on the command line
    """
    if filename_array is None:
        run_if_file_exists = False
        filename_array = [None for c in call_arrays]

    fclist = list(zip(filename_array, call_arrays))
    if scramble_run_order:
        shuffle(fclist)
    fnames, carrays = zip(*fclist)

    for i in tqdm(range(len(carrays))):
        call_array = carrays[i]
        if run_if_file_exists:
            file_missing = True
        else:
            fname = fnames[i]
            file_missing = not Path(fname).is_file()

        njobs = number_of_sbatch_jobs()
        nwait = 0

        while max_jobs < njobs:
            print(
                "waiting for jobs to complete, {:d} jobs running, waited {:.1f}min"
                .format(njobs, nwait * wait_if_full / 60.))
            sleep(wait_if_full)
            nwait += 1
            njobs = number_of_sbatch_jobs()

        if file_missing:
            call(call_array)
        sleep(wait_time)


def read_config(filename):
    if filename == "":
        print("You dont provide a config_path.")
        raise SystemExit

    with open(filename, "r") as f:
        try:
            config_data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            raise SystemExit
    return config_data


def detect_path(config_data):
    hostname = socket.gethostname()
    if "ci-connect" in hostname:
        path = config_data["OSG_path"]
        on_remote = True
    elif ("dali" in hostname) or ("midway" in hostname):
        path = config_data["midway_path"]
        on_remote = True
    else:
        path = "alea"
        on_remote = False
    return path, on_remote


def adapt_inference_object_config(config_data, wimp_mass, cache_dir=None):
    if "ll_config" in config_data.keys():
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn("Using ll_config is deprecated. Please use inference_object_config instead.", DeprecationWarning, stacklevel=2)
        if "inference_object_config" in config_data:
            raise ValueError("ll_config and inference_object_config are both present in config_data. Please remove one.")
        else:
            config_data["inference_object_config"] = config_data["ll_config"]
    inference_object_config = config_data["inference_object_config"]
    path, on_remote = detect_path(config_data)

    for ll_config in inference_object_config["likelihoods"]:
        analysis_space = ll_config["analysis_space"]
        new_analysis_space = []

        for element in analysis_space:
            for key, value in element.items():
                updated_element = (key,
                                   np.fromstring(value, dtype=float, sep=" "))
                new_analysis_space.append(updated_element)

        ll_config["analysis_space"] = new_analysis_space
        ll_config["default_source_class"] = locate(
            inference_object_config["default_source_class"])

        for uncertainty_name in ll_config["rate_parameter_uncertainties"]:
            this_uncertainty = ll_config["rate_parameter_uncertainties"][
                uncertainty_name]
            if isinstance(this_uncertainty, dict):
                if str(wimp_mass) in this_uncertainty.keys():
                    ll_config["rate_parameter_uncertainties"][
                        uncertainty_name] = this_uncertainty[str(wimp_mass)]
                else:
                    ll_config["rate_parameter_uncertainties"][
                        uncertainty_name] = this_uncertainty["default"]

        for source in ll_config["sources"]:
            if not on_remote:
                # now you submitted to the cluster
                source["templatename"] = os.path.join(
                    path, os.path.basename(source["templatename"]))
            else:
                # here you are on midway/dali or on OSG for "local" work
                source["templatename"] = os.path.join(path,
                                                      source["templatename"])
                if cache_dir is not None:
                    source["cache_dir"] = cache_dir
    return inference_object_config


def adapt_likelihood_config_for_blueice(likelihood_config: dict,
                                        template_folder: str) -> dict:
    """
    Adapt likelihood config to be compatible with blueice.

    Args:
        likelihood_config (dict): likelihood config dict
        template_folder (str): base folder where templates are located.
            If the folder starts with alea/, the alea folder is used as base.

    Returns:
        dict: adapted likelihood config
    """
    # if template folder starts with alea: get location of alea
    if template_folder.startswith("alea/"):
        import alea
        alea_dir = os.path.dirname(os.path.abspath(alea.__file__))
        template_folder = os.path.join(alea_dir, template_folder.replace("alea/", ""))

    likelihood_config["analysis_space"] = get_analysis_space(
        likelihood_config["analysis_space"])

    likelihood_config["default_source_class"] = locate(
        likelihood_config["default_source_class"])

    for source in likelihood_config["sources"]:
        source["templatename"] = os.path.join(template_folder,
                                              source["template_filename"])
    return likelihood_config


class VariationConvenience(object):
    """This class serves as a convenience wrapper to carry out cartesian products.
    The same functionality can be achieved with plain alea but this might
    come at the cost of complicated cartesian products written into 'parameters_to_zip'"""
    def __init__(self, parameters_to_vary: dict, parameters_to_zip: dict,
                 parameters_in_common: dict, parameters_as_wildcards: list,
                 generate_args_parameters: list, signature):
        self.parameters_to_vary = parameters_to_vary
        self.parameters_to_zip = parameters_to_zip
        self.paramters_in_common = parameters_in_common
        self.parameters_as_wildcards = parameters_as_wildcards
        self.generate_args_parameters = generate_args_parameters
        self.signature = signature

        if not bool(self.generate_args_parameters):
            print(
                "You do not provide a valid list of generate_args_parameters.")
            print("It cannot be empty.")
            raise SystemExit

        self.parameters_to_vary_in_generate_args = {}
        self.parameters_to_vary_in_signature = {}
        self.parameters_to_vary_in_likelihood = {}

        self.zip_basic_exists = False

    def split_parameters(self):
        for key, value in self.parameters_to_vary.items():
            if key in self.generate_args_parameters:
                self.parameters_to_vary_in_generate_args.update({key: value})
            elif key in self.signature.parameters:
                self.parameters_to_vary_in_signature.update({key: value})
            else:
                self.parameters_to_vary_in_likelihood.update({key: value})

    def genereate_alea_input(self):
        return itertools.product(*self.parameters_to_vary.values())

    def _create_zip_input(self):
        myproduct = itertools.product(*self.parameters_to_vary.values())
        self.zip_input = {i: [] for i in self.parameters_to_vary.keys()}
        for item in myproduct:
            for key, value in zip(self.zip_input.keys(), item):
                self.zip_input[key].append(value)
            else:
                self.zip_length = len(self.zip_input[key])
        self.zip_basic_exists = True

    def _create_generate_args_input(self):
        if not self.zip_basic_exists:
            self._create_zip_input()

        self.generate_args_to_zip = [
            dict() for number in range(self.zip_length)
        ]

        for par in self.parameters_to_vary_in_generate_args.keys():
            for arg, value in zip(self.generate_args_to_zip,
                                  self.zip_input[par]):
                arg.update({par: value})
        self.zip_input["generate_args"] = self.generate_args_to_zip

    def _create_likelihood_args_input(self):
        if not self.zip_basic_exists:
            self._create_zip_input()
        self._create_generate_args_input()
        self.inference_object_args = copy.deepcopy(
            self.zip_input["generate_args"])
        for item in self.inference_object_args:
            item["inference_object_config_overrides"] = {"likelihoods": [{}]}

        l_dict_to_append = []
        for par in self.parameters_to_vary_in_likelihood.keys():
            dict_to_append = dict()
            for idx, (arg, value) in enumerate(
                    zip(self.inference_object_args, self.zip_input[par])):
                arg.update({par: value})

                #  dict_to_append.update({par: value})
                #  # This part might need to be reworked once multiple likelihoods ar in use
                #  l_dict_to_append[idx].update(dict_to_append)
        for arg in self.inference_object_args:
            dict_to_add = dict()
            for key in arg.keys():
                if key == "inference_object_config_overrides":
                    continue
                else:
                    dict_to_add.update({key: arg[key]})
            arg["inference_object_config_overrides"]["likelihoods"][0] = dict_to_add
        self.zip_input["inference_object_args"] = self.inference_object_args

    def _propagate_guess(self):
        self.propagate_guess_to_zip = copy.deepcopy(self.generate_args_to_zip)
        self.zip_input["guess"] = self.propagate_guess_to_zip

    def combined_zip_input(self, propagate_guess=True):
        self.split_parameters()
        self._create_zip_input()
        self._create_generate_args_input()
        self._create_likelihood_args_input()
        if propagate_guess:
            self._propagate_guess()


def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def compute_variations(parameters_to_vary,
                       parameters_to_zip,
                       parameters_in_common,
                       special_parameters=None,
                       silent=False):
    """
    if parameters are defined in multiple places the order or precedence is:
    1. parameters_to_zip takes priority over
    2. parameters_to_vary takes priority over
    3. parameters_in_common

    """
    # Check that the signal_rate_multiplier is not varied when signal_expectation is not None
    ptz_srm = parameters_to_zip.get("generate_args", [{}])[0].get("signal_rate_multiplier", None)
    ptv_srm = parameters_to_vary.get("generate_args", [{}])[0].get("signal_rate_multiplier", None)
    ptz_se = parameters_to_zip.get("signal_expectation", None)
    ptv_se = parameters_to_vary.get("signal_expectation", None)
    pic_se = parameters_in_common.get("signal_expectation", None)
    assert (ptz_se is None and ptv_se is None and pic_se is None) or (ptz_srm is None and ptv_srm is None), "signal_rate_multiplier cannot be varied when signal_expectation is not None"

    varied_dicts = compute_parameters_to_vary(
        parameters_to_vary=parameters_to_vary)
    zipped_dicts = compute_parameters_to_zip(
        parameters_to_zip=parameters_to_zip, silent=silent)
    combined_variations = list(itertools.product(varied_dicts, zipped_dicts))
    if special_parameters is None:
        special_parameters = [
            "generate_args", "livetime", "inference_object_args"
        ]

    if len(combined_variations) > 0:
        keys_are_shared = bool(
            set(combined_variations[0][0]) & set(combined_variations[0][1]))
        if keys_are_shared:
            if not silent:
                print("There are shared keys between parameters")
            shared_parameters = list(
                set(combined_variations[0][0]).intersection(
                    combined_variations[0][1]))
            if not silent:
                print(shared_parameters)
                print("Looking for problems now...")
            problematic_parameters = []
            for parameter in shared_parameters:
                if parameter not in special_parameters:
                    problematic_parameters.append(parameter)

            if len(problematic_parameters) == 0:
                if not silent:
                    print(
                        "You still need to watch out that everything is correct but only special_parameters are shared"
                    )
            else:
                if len(problematic_parameters) > 1:
                    message = " ".join(problematic_parameters) + " are shared."
                else:
                    message = " ".join(problematic_parameters) + " is shared."
                raise Exception(message)

    merged_combinations = []
    for variation, zipped in tqdm.tqdm(combined_variations, disable=silent):
        pic = copy.deepcopy(parameters_in_common)
        mergedeep.merge(pic, variation, zipped)
        merged_combinations.append(pic)
    else:
        return merged_combinations


def flatten_nested_variables(merged_combinations):
    copy_of_merged_combinations = copy.deepcopy(merged_combinations)
    for idx, combination in enumerate(merged_combinations):
        if "generate_args" in copy_of_merged_combinations[idx].keys():
            for key, value in combination["generate_args"].items():
                if key in copy_of_merged_combinations[idx].keys(
                ) and value != copy_of_merged_combinations[idx][key]:
                    raise Exception(
                        "You are overwriting function_args without knowing what is going on!"
                    )
                elif key == "signal_rate_multiplier" and combination["signal_expectation"] is not None:
                    continue
                else:
                    copy_of_merged_combinations[idx][key] = value
        if "livetime" in combination.keys():
            # check if combination["livetime"] is a list or a single value
            if isinstance(combination["livetime"], list):
                for idx_lt, value in enumerate(combination["livetime"]):
                    if "livetime_" + str(idx_lt) in combination.keys():
                        raise Exception(
                            "You are overwriting function_args without knowing what is going on!"
                        )
                    copy_of_merged_combinations[idx]["livetime_" +
                                                     str(idx_lt)] = value
            else:
                if "livetime" in copy_of_merged_combinations[idx].keys(
                ) and combination["livetime"] != copy_of_merged_combinations[
                        idx]["livetime"]:
                    raise Exception(
                        "You are overwriting function_args without knowing what is going on!"
                    )
                copy_of_merged_combinations[idx]["livetime"] = combination[
                    "livetime"]
    return copy_of_merged_combinations


def flatten_function_args(combination, function_args):
    if "generate_args" in combination.keys():
        for key, value in combination["generate_args"].items():
            if key in function_args:
                raise Exception(
                    "You are overwriting function_args without knowing what is going on!"
                )
            elif key == "signal_rate_multiplier" and combination.get("signal_expectation", None) is not None:
                continue
            else:
                function_args[key] = value
    if "livetime" in combination.keys():
        if isinstance(combination["livetime"], list):
            for idx, value in enumerate(combination["livetime"]):
                if "livetime_" + str(idx) in function_args:
                    raise Exception(
                        "You are overwriting function_args without knowing what is going on!"
                    )
                function_args["livetime_" + str(idx)] = value
    return function_args


def compute_parameters_to_vary(parameters_to_vary):
    for k in copy.deepcopy(parameters_to_vary):
        if isinstance(parameters_to_vary[k], dict):
            # allows variations inside of dicts
            parameters_to_vary[k] = [
                item for item in dict_product(parameters_to_vary[k])
            ]
        else:
            parameters_to_vary[k] = parameters_to_vary[k]

    # these are the variations of parameters_to_vary
    cartesian_product = itertools.product(*parameters_to_vary.values())
    parameter_names = parameters_to_vary.keys()

    variations_to_return = []

    for variation in cartesian_product:
        parameter_configuration = {
            key: value
            for key, value in zip(parameter_names, variation)
        }
        variations_to_return.append(parameter_configuration)

    if len(variations_to_return) == 0:
        return [{}]
    else:
        return variations_to_return


def compute_parameters_to_zip(parameters_to_zip, silent=False):
    ptz = copy.deepcopy(parameters_to_zip)
    # check that inputs have the same length
    # 1. get all lists
    all_lists = []
    for key, value in ptz.items():
        if isinstance(value, list):
            all_lists.append(value)
        elif isinstance(value, dict):
            l_dicts = []
            if len(value) == 1:
                key_inner, item = list(value.keys())[0], list(value.values())[0]
                if isinstance(item, list):
                    all_lists.append(item)
                    expanded_dicts = [{
                        key_inner: list_value
                    } for list_value in item]
                    ptz[key] = expanded_dicts
                    # manipulate dict to be present 4 times
                else:
                    raise NotImplementedError(
                        "parameters_to_zip not implemented for dict with values of type("
                        + str(type(value)) + ")")
            else:
                ptz[key] = list(dict_product(value))
                all_lists.append(ptz[key])
        else:
            raise NotImplementedError(
                "parameters_to_zip not implemented for type(" + type(value) +
                ")")

    # 2. check that all values have the same length
    if len(all_lists) > 0:
        it = iter(all_lists)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            raise ValueError('not all lists have same length!')

    varied_dicts_zip = []
    for values in zip(*ptz.values()):
        this_dict = {key: value for key, value in zip(ptz.keys(), values)}
        varied_dicts_zip.append(this_dict)

    if len(all_lists) > 0:
        if len(all_lists[0]) != len(varied_dicts_zip):
            raise Exception(
                "Zipping failed. You probably escaped checking with a special case."
            )
    else:
        if not silent:
            print(
                "Cannot check sanity of zip - better provide a list like, var: [1,2,3]"
            )

    if len(varied_dicts_zip) == 0:
        return [{}]
    else:
        return varied_dicts_zip


def manipulate_toy_data(statistical_model,
                        data_sets,
                        sources_to_find,
                        plot=False,
                        output_dir=None):
    import matplotlib.pyplot as plt
    smearing_pars = {
        '[0, 10]': np.array([19.16806452, 2.80622276, 0.20968269]),
        '[10, 20]': np.array([21.54000255, 2.98114261, 0.15294833])
    }
    l_modified_data = []
    for idx, (data, ll) in enumerate(zip(data_sets, statistical_model.lls)):
        modified_data = copy.deepcopy(data)
        masker = ""
        for source_name in sources_to_find:
            for source_idx, source in enumerate(ll.base_model.sources):
                if source.name == source_name:
                    masker += f'(data["source"] == {source_idx})|'

        if len(masker) > 0:
            rg_and_rgx = eval(masker[:-1])
        else:
            rg_and_rgx = np.array([True] * len(data))

        if idx > 1:
            continue
        for cs1_range, popt in smearing_pars.items():
            xspace = np.linspace(0, 4, 100)
            this_range = eval(cs1_range)
            data_mask = rg_and_rgx & (data["cs1"] > this_range[0]) & (
                data["cs1"] < this_range[1])
            nevents = np.sum(data_mask)
            print(f"cs1 range: {cs1_range}, nevents: {nevents}")
            modified_RG_and_RGX = sps.norm.rvs(loc=popt[1],
                                               scale=popt[2],
                                               size=nevents)
            modified_data["log10cs2"][data_mask] = modified_RG_and_RGX

            if plot:
                fig, ax = plt.subplots()
                title = f"{ll_names[idx]}, cs1 range: {cs1_range}PE"
                ax.set_title(title)
                ax.scatter(data[~data_mask]["cs1"],
                           data[~data_mask]["log10cs2"],
                           color="blue")
                ax.scatter(data[data_mask]["cs1"],
                           data[data_mask]["log10cs2"],
                           color="red",
                           label=f"RG(X) < {this_range[1]} PE")
                ax.axvline(this_range[1], color="red")
                ax.set_xlim(0, 100)

                ax.scatter(data[data_mask]["cs1"],
                           modified_RG_and_RGX,
                           color="orange",
                           label=f"RG(X) from AmBe")
                ax.legend()

                fig, ax = plt.subplots()
                print(len(modified_RG_and_RGX), len(data[data_mask]))
                binning = np.linspace(2, 4, 20)
                ax.hist(modified_RG_and_RGX,
                        label="from AmBe",
                        bins=binning,
                        alpha=0.5)
                ax.hist(data[data_mask]["log10cs2"],
                        bins=binning,
                        label="Template",
                        alpha=0.5)

                median = np.median(modified_RG_and_RGX)
                label = f"AmBe median at {median:.2f}"
                ax.axvline(np.median(modified_RG_and_RGX),
                           label=label,
                           color="C0")
                ax.axvline(np.median(data[data_mask]["log10cs2"]),
                           label="Template median",
                           color="C1")
                ax.set_xlim(2, 4)
                ax.set_xlabel("log10cs2")
                ax.legend()
                plt.show()
        else:
            l_modified_data.append(modified_data)
    else:
        return l_modified_data


def _divide_fit_result(ll, fit_result):
    l_parameters = []
    for parameter in ll.rate_parameters:
        this_par = parameter + "_rate_multiplier"
        if this_par in fit_result.keys():
            l_parameters.append(this_par)

    for parameter in ll.shape_parameters:
        if parameter in fit_result.keys():
            l_parameters.append(parameter)

    fit_result_to_return = copy.deepcopy(fit_result)
    for par in copy.deepcopy(fit_result_to_return).keys():
        if par not in l_parameters:
            del fit_result_to_return[par]

    return fit_result_to_return


def get_log_prior_from_file(filename, parameter):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    if parameter not in data.keys():
        raise ValueError(f"Parameter {parameter} not in {filename}")

    parameter_to_return = data[parameter]
    if isinstance(parameter_to_return, dict):
        if "uncertainty" not in parameter_to_return.keys():
            raise Exception("No uncertainty in file")
        if "central_value" not in parameter_to_return.keys():
            raise Exception("No central_value in file")
    return parameter_to_return


def _get_generate_args_from_toyfile(toydata_file):
    datasets_array, _ = toydata_from_file(toydata_file)
    # check that all generate_args in the toyfile are the same
    generate_args_list = [toy_data[-1] for toy_data in datasets_array]
    all_generate_args_equal = all(generate_args_list[0] == generate_args
                                  for generate_args in generate_args_list)
    if not all_generate_args_equal:
        raise ValueError("generate_args in toyfile are not all the same")
    return structured_array_to_dict(generate_args_list[0])


def get_analysis_space(analysis_space: dict) -> list:
    eval_analysis_space = []

    for element in analysis_space:
        for key, value in element.items():
            if value.startswith("np."):
                eval_element = (key, eval(value))
            else:
                eval_element = (key,
                                np.fromstring(value,
                                              dtype=float,
                                              sep=" "))
            eval_analysis_space.append(eval_element)
    return eval_analysis_space
