import re
import warnings
import os
from copy import deepcopy
from pydoc import locate  # to lookup inferenceObject class

import alea.utils as utils
import h5py
import numpy as np
import scipy.stats as sps
from alea._plotting import pdf_plotter
from alea.simulators import BlueiceDataGenerator
from blueice.inference import bestfit_scipy, one_parameter_interval
from blueice.likelihood import LogLikelihoodSum
from inference_interface import (dict_to_structured_array,
                                 structured_array_to_dict, toydata_from_file,
                                 toydata_to_file)
from scipy.interpolate import interp1d

import logging
logging.basicConfig(level=logging.INFO, force=True)

minimize_kwargs = {'method': 'Powell', 'options': {'maxiter': 10000000}}


class InferenceObject:
    """
    Args:
        wimp_mass (int): WIMP mass in GeV.
            Overwrites wimp_mass in every likelihood term.
        livetime (float): Livetime of the data. Unit depends on templates.
        config_path (str): path to yaml-configfile which define the likelihood
        generate_args (dict): dictionary which is used to created toyMC data.
        confidence_level (float, optional): Confidence Level. Defaults to 0.9.
        toydata_file: path to file containing toydata,
            defaults to None
        toydata_mode: "none", "write" or "read".
            - if "write": simulate interpolated ll and write toydata to 'toydata_file'
            - if "read": read toydata from 'toydata_file'
            - if "none": simulate interpolated ll
            - if "pass": not generateing toys
            Defaults to "write"
        limit_threshold (str, optional): path to file from which the thresholds are read.
            If None use asymptotics.
            defaults to None
        confidence_level (float, optional): confidence level for which sensitivity is computed.
            defaults to 0.9
        inference_object_config_overrides (dict, optional): Dict of values that are
            overridden in the inference_object_config dict provided by config file
            defaults to {}
    """
    def __init__(self,
                 wimp_mass,
                 livetime,
                 config_path,
                 threshold_key=None,
                 generate_args=None,
                 toydata_file=None,
                 toydata_mode="write",
                 limit_threshold=None,
                 confidence_level=0.9,
                 inference_object_config_overrides=None,
                 cache_dir=None,
                 **kwargs):
        self.wimp_mass = wimp_mass
        if isinstance(livetime, float):
            self.livetime = [livetime]
        else:
            self.livetime = livetime

        if inference_object_config_overrides is None:
            inference_object_config_overrides = {}

        self.config_path = config_path
        self.default_multiplier = None
        if generate_args is None:
            self.generate_args = {}
        elif generate_args == "default":
            print("Setting default_multiplier to 1")
            self.default_multiplier = 1
            self.generate_args = {}
        else:
            self.generate_args = generate_args

        if toydata_mode == "read":
            generate_args_from_toydata = utils._get_generate_args_from_toyfile(toydata_file)      
            if not ((self.generate_args == generate_args_from_toydata) or (self.generate_args == {})):
                warnings.warn("The generate_args you provided are not the same as the ones in the toydata file."
                              + " Using the parsed generate_args.", stacklevel=2)
            else:
                self.generate_args = generate_args_from_toydata

        self.inference_object_config_overrides = inference_object_config_overrides
        self.limit_threshold = limit_threshold
        self.confidence_level = confidence_level
        self.toydata_file = toydata_file
        self.toydata_mode = toydata_mode
        self.kwargs = kwargs
        self.threshold_key = threshold_key
        self.cache_dir = cache_dir

        self.variables_to_read = dict()
        self.variables_to_read["wimp_mass"] = self.wimp_mass
        self.variables_to_read["livetime"] = self.livetime

        for idx, livetime in enumerate(self.livetime):
            setattr(self, "livetime_{}".format(idx), livetime)

        config_data = utils.read_config(config_path)
        self.config_data = config_data
        inference_object_config = utils.adapt_inference_object_config(config_data, wimp_mass=wimp_mass, cache_dir=self.cache_dir)

        self.limit_threshold_function = None

        overwrite_items = inference_object_config_overrides.get("likelihoods", [])
        if overwrite_items:  # check if list is not empty
            del inference_object_config_overrides["likelihoods"]

        if self.livetime is not None and len(inference_object_config["likelihoods"]) != len(
                self.livetime):
            raise Exception(
                "You try to set a livetime for a combined likelihood. You need to pass as many livetimes as there are likelihoods."
            )

        for idx, ll_config in enumerate(inference_object_config["likelihoods"]):
            ll_config["wimp_mass"] = self.wimp_mass
            if self.livetime is not None:
                if self.livetime[idx] is not None:
                    ll_config["livetime_days"] = self.livetime[idx]
            if idx < len(overwrite_items):
                ll_config.update(overwrite_items[idx])
        inference_object_config.update(inference_object_config_overrides)
        self.inference_object_config = inference_object_config

        input_files = []
        for ll_config in self.inference_object_config["likelihoods"]:
            for source in ll_config["sources"]:
                input_files.append(source["templatename"])
            for uncertainty_name, uncertainty_setting in ll_config["shape_parameters"].items():
                if isinstance(uncertainty_setting, dict) and "path_to_prior" in uncertainty_setting.keys():
                    path, _ = utils.detect_path(self.config_data)
                    input_files.append(os.path.join(path, uncertainty_setting["path_to_prior"]))
        self.input_files = input_files

        lls = []
        dataset_names = []
        constraint_term_list = []  # will be filled with keys of likelihood parameters to check if a constraint term is added multiple times to the likelihood.
        binned = []
        for ll_config in self.inference_object_config["likelihoods"]:
            if ll_config["likelihood_type"] == "blueice.likelihood.BinnedLogLikelihood":
                binned.append(True)
            elif ll_config["likelihood_type"] == "blueice.likelihood.UnbinnedLogLikelihood":
                binned.append(False)
            else:
                raise NotImplementedError
            likelihood_object = locate(ll_config["likelihood_type"])
            ll = likelihood_object(ll_config)
            for parameter, relative_uncertainty in ll_config[
                    "rate_parameter_uncertainties"].items():
                if relative_uncertainty is not None:
                    constraint_already_set = parameter in constraint_term_list
                    if constraint_already_set:
                        warnings.warn(f"The constraint term for {parameter} rate is added by multiple parts of the likelihood.")
                    constraint_term_list.append(parameter)
                    print("Setting rate parameter uncertainty for {} to {}".
                          format(parameter, relative_uncertainty))
                    parameter_central_value = self.generate_args.get(
                        parameter + "_rate_multiplier",
                        self.default_multiplier)
                    self.generate_args[
                        parameter +
                        "_rate_multiplier"] = parameter_central_value
                    if parameter_central_value is None:
                        print("Unknown parameter", parameter)
                        print(
                            "generate_args and rate_parameter_uncertainties seem incompatible"
                        )
                        print("generate_args are:", self.generate_args)
                        print("rate_parameter_uncertainties are:",
                              ll_config["rate_parameter_uncertainties"])
                        raise Exception
                    ll.add_rate_parameter(parameter,
                                          log_prior=sps.norm(
                                              1, parameter_central_value *
                                              relative_uncertainty).logpdf)
                else:
                    ll.add_rate_parameter(parameter)

            # not tested!
            for parameter, values in ll_config["shape_parameters"].items():
                dict_to_set = {}
                dict_to_set[parameter] = deepcopy(values)
                if isinstance(values, dict) and "log_prior" in values.keys():
                    constraint_already_set = parameter in constraint_term_list
                    if constraint_already_set:
                        warnings.warn(f"The constraint term for {parameter} shape is added by multiple parts of the likelihood.")
                    constraint_term_list.append(parameter)
                    # this_log_prior = eval(values["log_prior"])
                    # values["log_prior"] = this_log_prior
                    this_log_prior = sps.norm(loc=values["log_prior"]["loc"], scale=values["log_prior"]["scale"]).logpdf
                    # values["log_prior"] = this_log_prior
                    dict_to_set[parameter]["log_prior"] = this_log_prior
                elif isinstance(values, dict) and "path_to_prior" in values.keys():
                    constraint_already_set = parameter in constraint_term_list
                    if constraint_already_set:
                        warnings.warn(f"The constraint term for {parameter} shape is added by multiple parts of the likelihood.")
                    constraint_term_list.append(parameter)
                    filename_expander = values.get("filename_expander", {})
                    transformed_expander = {}
                    for string_name, value_to_get in filename_expander.items():
                        transformed_expander[string_name] = getattr(self, value_to_get)

                    BASEPATH, on_remote = utils.detect_path(self.config_data)
                    if on_remote:
                        file_to_open = os.path.join(BASEPATH, values["path_to_prior"].format(**transformed_expander))
                    else:
                        # ensures that file is expected in top-level
                        file_to_open = os.path.join(BASEPATH, os.path.basename(values["path_to_prior"].format(**transformed_expander)))
                    log_prior_from_file = utils.get_log_prior_from_file(file_to_open, parameter=parameter)
                    if isinstance(log_prior_from_file, dict):
                        # make the function yourself from what is stored in file
                        values["log_prior"] = {}
                        values["log_prior"]["loc"] = log_prior_from_file["central_value"]
                        values["log_prior"]["scale"] = log_prior_from_file["uncertainty"]
                        dict_to_set[parameter]["log_prior"] = sps.norm(log_prior_from_file["central_value"], log_prior_from_file["uncertainty"]).logpdf
                    else:
                        # then the log_prior is a function
                        values["log_prior"] = log_prior_from_file
                        dict_to_set[parameter]["log_prior"] = log_prior_from_file

                    if "path_to_prior" in dict_to_set[parameter].keys():
                        del dict_to_set[parameter]["path_to_prior"]
                    if "filename_expander" in dict_to_set[parameter].keys():
                        del dict_to_set[parameter]["filename_expander"]

                if isinstance(values, list):
                    ll.add_shape_parameter(parameter, dict_to_set[parameter])
                else:
                    ll.add_shape_parameter(parameter, **dict_to_set[parameter])

            ll.prepare()

            dtype = [(key[0], float) for key in ll_config["analysis_space"]]
            dummy_data = np.zeros(1, dtype)
            for key in ll_config["analysis_space"]:
                dummy_data[key[0]] = (key[1][-1] - key[1][0]) / 2

            #  The above used to be: might be useful for debugging
            #  dtype = [("cs1", float), ("logcs2b", float)]
            #  dummy_data = np.zeros(1, dtype)
            #  dummy_data["cs1"] = 50.
            #  dummy_data["logcs2b"] = 2.
            ll.set_data(dummy_data)
            lls.append(ll)
            dataset_names.append(ll_config["dataset_name"])
        dataset_names += inference_object_config["dataset_names"]
        self.dataset_names = dataset_names
        self.lls = lls
        self.binned = binned
        self.pdf_plotters = [pdf_plotter(ll) for ll in self.lls]

        self.signal_expectations = None
        self.thresholds = None
        self.nominal_signal_expectation = None
        self.datasets_array = None
        self.set_toy_reading_mode(toydata_mode=self.toydata_mode, toydata_file=self.toydata_file)
        self.ll = LogLikelihoodSum(self.lls, likelihood_weights=self.kwargs.get("likelihood_weights", None))

        self.promote_generate_args()

    def set_toy_reading_mode(self, toydata_mode, toydata_file):
        self.toydata_mode = toydata_mode
        self.toydata_file = toydata_file
        if toydata_mode == "none":
            self.rgs = [BlueiceDataGenerator(ll, binned=b) for ll, b in zip(self.lls, self.binned)]
            if hasattr(self, "datasets_array"):
                del self.datasets_array
            if hasattr(self, "toydata_index"):
                del self.toydata_index
        elif toydata_mode == "write":
            self.rgs = [BlueiceDataGenerator(ll, binned=b) for ll, b in zip(self.lls, self.binned)]
            self.datasets_array = []
            if hasattr(self, "toydata_index"):
                del self.toydata_index
        elif toydata_mode == "read":
            dataset_names = self.load_toydata(toydata_file)
            assert self.dataset_names == dataset_names
            self.toydata_index = 0
        elif toydata_mode == "pass":
            self.toydata_index = 0

        else:
            print("Unknown toydata_mode {toydata_mode}".format(
                toydata_mode=self.toydata_mode))
            print("Allowed modes are none, write, read, pass")
            raise SystemExit

    def load_toydata(self, toydata_file):
        self.datasets_array, dataset_names = toydata_from_file(
            toydata_file)
        return dataset_names

    def full_ll(self, **kwargs):
        for key in kwargs:
            if key not in self.get_parameter_list():
                raise Exception("Unknown parameter {}".format(key))
        return self.ll(**kwargs)

    def build_threshold_key(self, threshold_key=None):
        if self.limit_threshold is None:
            raise Exception(
                "You need to set a limit_threshold, a file where the thresholds are stored."
            )

        if threshold_key is None:
            with h5py.File(self.limit_threshold, "r") as f:
                threshold_key_pattern = f.attrs["threshold_key_pattern"]

            variables = re.findall("\{(.*?)\}", threshold_key_pattern)
            variables = [var.split(":")[0] for var in variables]

            for var in variables:
                if var not in dir(self):
                    raise Exception(
                        "Variable {} is not in the list of variables to read".
                        format(var))
            threshold_key = threshold_key_pattern.format(**self.__dict__)
            threshold_key = threshold_key.replace("\"", "")
        else:
            # print a warning with the warning module
            warnings.warn(
                "You are using a custom threshold_key. This is not recommended, as the thresholds and parameters most likely mismatch."
            )

        self.check_if_key_exists(threshold_key)
        self.threshold_key = threshold_key

    def check_if_key_exists(self, threshold_key):
        with h5py.File(self.limit_threshold, "r") as f:
            if threshold_key not in f:
                raise Exception(
                    "Threshold key \n{}\n is not in the list of thresholds in the file {}"
                    .format(threshold_key, self.limit_threshold) +
                    "\nAvailable keys are:" + "\n".join(f.keys()))

    def _set_limit_threshold_function(self):
        if self.limit_threshold is None:
            print("No limit_threshold set. Using asymptotics.")
            limit_threshold_function = lambda x, dummy: sps.chi2(1).isf(0.1)
            self.signal_expectations = None
            self.thresholds = None
            self.nominal_signal_expectation = None

            return limit_threshold_function
        else:
            print(
                "loading limit_threshold {:s}, confidence level {:.2f}".format(
                    self.limit_threshold, self.confidence_level))

            self.signal_expectations, self.thresholds, self.nominal_signal_expectation = utils.read_neyman_threshold_update(
                self.limit_threshold,
                self.threshold_key,
                confidence_level=self.confidence_level)

            print("loaded threshold")
            print("signal_expectations", self.signal_expectations)
            print("thresholds", self.thresholds)
            print("nominal_signal_expectation",
                  self.nominal_signal_expectation)
            ltf = interp1d(self.signal_expectations /
                           self.nominal_signal_expectation,
                           self.thresholds,
                           bounds_error=False,
                           fill_value=sps.chi2(1).isf(0.1))

            def limit_threshold_function(x, cl):
                return ltf(x)

            return limit_threshold_function

    def assign_data(self, datas):
        for data, ll in zip(datas, self.lls):
            ll.set_data(data)

    def simulate_and_assign_data(self, generate_args):
        logging.debug("simulate_and_assign_data with generate_args", generate_args)
        self._check_generate_args(generate_args=generate_args)
        datas = [rg.simulate(**generate_args) for rg in self.rgs]
        self.assign_data(datas)
        return datas

    def simulate_and_assign_measurements(self, generate_args):
        self._check_generate_args(generate_args=generate_args)
        ancillary_measurements = []
        for ll_config in self.inference_object_config["likelihoods"]:
            ret = dict()
            # loop over rate parameters
            for parameter_name, parameter_uncert in ll_config[
                    "rate_parameter_uncertainties"].items():
                if parameter_uncert is not None:
                    parameter_mean = generate_args.get(
                        parameter_name + "_rate_multiplier", 1)
                    parameter_meas = max(
                        0,
                        sps.norm(parameter_mean,
                                 parameter_mean * parameter_uncert).rvs())
                    ret[parameter_name + "_rate_multiplier"] = parameter_meas
            # loop over shape parameters
            for parameter_name, parameter_uncert in ll_config["shape_parameters"].items():
                if "log_prior" in parameter_uncert.keys():
                    parameter_mean = generate_args.get(parameter_name)
                    parameter_meas = sps.norm(parameter_mean,
                                 parameter_uncert["log_prior"]["scale"]).rvs()
                    ret[parameter_name] = parameter_meas

            ancillary_measurements.append(ret)
        self.assign_measurements(ancillary_measurements=ancillary_measurements,
                                 generate_args=generate_args)
        return ancillary_measurements

    def assign_measurements(self, ancillary_measurements, generate_args):
        if isinstance(ancillary_measurements, dict):
            ancillary_measurements = [ancillary_measurements] * len(
                self.lls)
        for ancillary_measurement, ll_config, ll in zip(ancillary_measurements, self.inference_object_config["likelihoods"], self.lls):
            for parameter_name, parameter_uncert in ll_config[
                    "rate_parameter_uncertainties"].items():
                if parameter_uncert is not None:
                    parameter_mean = generate_args.get(
                        parameter_name + "_rate_multiplier", 1)
                    parameter_meas = ancillary_measurement[parameter_name +
                                                           "_rate_multiplier"]
                    logging.debug(f"Assign measurement for {parameter_name}")
                    logging.debug(f"Normal distribution with mean {parameter_meas:.5f} and width {parameter_mean * parameter_uncert:.5f}.")
                    ll.rate_parameters[parameter_name] = sps.norm(
                        parameter_meas,
                        parameter_mean * parameter_uncert).logpdf

        for ancillary_measurement, ll_config, ll in zip(ancillary_measurements, self.inference_object_config["likelihoods"], self.lls):
            for parameter_name, parameter_uncert in ll_config[
                    "shape_parameters"].items():
                if "log_prior" in parameter_uncert.keys():
                    parameter_meas = ancillary_measurement[parameter_name]
                    shape_options = list(deepcopy(ll.shape_parameters[parameter_name]))
                    shape_options[1] = sps.norm(
                        parameter_meas,
                        parameter_uncert["log_prior"]["scale"]).logpdf
                    ll.shape_parameters[parameter_name] = tuple(shape_options)

    def llr(self,
            extra_args=None,
            extra_args_null=None,
            guess=None):

        if extra_args is None:
            extra_args = {}
        if guess is None:
            guess = {}
        if extra_args_null is None:
            extra_args_null = {"signal_rate_multiplier": 0.}
        extra_args_null_total = deepcopy(extra_args)
        extra_args_null_total.update(extra_args_null)
        res1, llval1 = bestfit_scipy(self.ll,
                                     guess=guess,
                                     minimize_kwargs=minimize_kwargs,
                                     **extra_args)
        res0, llval0 = bestfit_scipy(self.ll,
                                     guess=guess,
                                     minimize_kwargs=minimize_kwargs,
                                     **extra_args_null_total)
        return 2. * (llval1 - llval0), llval1, res1, llval0, res0

    def confidence_interval(self,
                            llval_best,
                            extra_args=None,
                            guess=None,
                            parameter_name="signal_rate_multiplier",
                            two_sided=True):
        if guess is None:
            guess = {}
        if extra_args is None:
            extra_args = {}

        #the confidence interval computation looks in a bounded region-- we will say that we will not look for larger than 300 signal events
        rate_multiplier_max = 10000. / self.get_mus(**extra_args).get(
            parameter_name.replace("_rate_multiplier", ""), 1.)
        rate_multiplier_min = 0.
        logging.debug("rate_multiplier_max: " + str(rate_multiplier_max))

        if self.limit_threshold_function is None:
            self.limit_threshold_function = self._set_limit_threshold_function(
            )

        dl = -1 * np.inf
        ul = one_parameter_interval(self.ll,
                                    parameter_name,
                                    rate_multiplier_max,
                                    bestfit_routine=bestfit_scipy,
                                    minimize_kwargs=minimize_kwargs,
                                    t_ppf=self.limit_threshold_function,
                                    guess=guess,
                                    **extra_args)
        if two_sided:
            extra_args_null = deepcopy(extra_args)
            extra_args_null[parameter_name] = rate_multiplier_min

            res_null, llval_null = bestfit_scipy(
                self.ll,
                guess=guess,
                minimize_kwargs=minimize_kwargs,
                **extra_args_null)
            llr = 2. * (llval_best - llval_null)
            if llr <= self.limit_threshold_function(rate_multiplier_min, 0):
                dl = rate_multiplier_min
            else:
                dl = one_parameter_interval(
                    self.ll,
                    parameter_name,
                    rate_multiplier_min,
                    kind="lower",
                    bestfit_routine=bestfit_scipy,
                    minimize_kwargs=minimize_kwargs,
                    t_ppf=self.limit_threshold_function,
                    guess=guess,
                    **extra_args)
        return dl, ul

    def generate_toydata(self, generate_args):
        datas = self.simulate_and_assign_data(generate_args=generate_args)
        ancillary_measurements = self.simulate_and_assign_measurements(
            generate_args=generate_args)
        if self.toydata_mode == "write":
            all_keys = []
            ancillary_measurements_to_store = {}
            for ancillary_measurement in ancillary_measurements:
                for key, value in ancillary_measurement.items():
                    all_keys.append(key)
                    ancillary_measurements_to_store[key] = value
            if len(all_keys) != len(set(all_keys)):
                raise ValueError("WARNING: some keys are repeated in the ancillary measurements.")

            datas.append(dict_to_structured_array(ancillary_measurements_to_store))
            if 0 < len(generate_args):
                datas.append(dict_to_structured_array(generate_args))
            else:
                datas.append(dict_to_structured_array({"alldefault": 0}))

            self.datasets_array.append(datas)
        return datas, ancillary_measurements

    def read_toydata(self, generate_args):
        datas = self.datasets_array[self.toydata_index]
        self.toydata_index += 1
        self.assign_data(datas)
        #print("ancillary measurement is", datas[-2], datas[-2].dtype)
        #print("ancillary measurement length",len(datas[-2]))
        ancillary_measurements = structured_array_to_dict(datas[-2])
        #print("ancillary measurement",ancillary_measurements)
        self.assign_measurements(ancillary_measurements, generate_args)

    def _check_generate_args(self, generate_args):
        parameters = self.get_parameter_list()
        for par_to_pop in ["ll", "dl", "ul"]:
            if par_to_pop in parameters:
                parameters.remove(par_to_pop)
        generate_args_present = sorted(list(generate_args.keys())) == sorted(parameters)
        if not generate_args_present:
            error_msg = ""
            for parameter in parameters:
                if parameter not in generate_args.keys():
                    error_msg += f"necessary parameter {parameter} not in generate_args\n"

            for parameter in generate_args.keys():
                if parameter not in parameters:
                    error_msg += f"parameter {parameter} defined in generate_args not in parameter list\n"
            raise ValueError(error_msg)


    def toy_simulation(self,
                       generate_args=None,
                       extra_args=None,
                       guess=None,
                       compute_confidence_interval=False,
                       confidence_interval_args=None,
                       propagate_guess=True):
        """Read in next toy dataset (toydata_mode == "read") or simulate
        toy dataset with given `generate_args` (toydata_mode == "write" or "none")
        and assign data and measurements.
        A log likelihood fit of the dataset is performed via
        `blueice.inference.bestfit_scipy` for each extra arg in `extra_args`.


        Args:
            generate_args (dict, optional): Dict of rate, signal and
                shape parameters. Defaults to {}.
            extra_args (list, optional): each toyMC is fit with each
                extra_arg. Defaults to [{},{"signal_rate_multiplier":0.}].
            guess (dict, optional): Guess passed to fitter.
                Defaults to {"signal_rate_multiplier":0.}.
            compute_confidence_interval (bool, optional): if True, the function
                will compute confidence intervals on the
                signal_rate_multiplier, for the _first_ set of extra_args.
                Defaults to False.
            confidence_interval_args (dict, optional): If
                `compute_confidence_interval` is True, these kwargs are passed
                to self.confidence_interval. Defaults to {}.
            propagate_guess (bool, optional): If True, the previous fit results
                (except the values specified in guess) are used as a guess,
                defaults to True

        Returns:
            ress: List of ML fit results for all extra_args
        """
        logging.debug("toy_simulation with generate_args", generate_args)
        if generate_args is None:
            generate_args = {}

        if extra_args is None:
            extra_args=[{}, {
                "signal_rate_multiplier": 0.
            }]
        if guess is None:
            guess={"signal_rate_multiplier": 0.}
        if confidence_interval_args is None:
            confidence_interval_args={}
        if self.toydata_mode == "read":
            self.read_toydata(generate_args)
        elif self.toydata_mode == "pass":
            print("Not generating toy data")
        else:
            self.generate_toydata(generate_args=generate_args)
        for i, ll in enumerate(self.lls):
            logging.debug(f"toy data in ll with index {i} has length " +  str(len(ll._data)))
            logging.debug(f"source names: " +  str(ll.source_name_list))
            for j, name in enumerate(ll.source_name_list):
                logging.debug(f"number of simulated {name} events: " +  str(len(ll._data[ll._data["source"] == j])))
            logging.debug(f"rate_parameters in ll with index {i}: " + str(ll.rate_parameters))
            logging.debug(f"shape_parameters in ll with index {i}: " + str(ll.shape_parameters))

        ress = []
        extra_args_runs = extra_args
        if type(extra_args_runs) is dict:
            extra_args_runs = [extra_args_runs]

        previous_fit = {}
        logging.debug("Propagate guess: " + str(propagate_guess))
        for extra_args_run in extra_args_runs:
            guess_run = {}
            if propagate_guess:
                guess_run.update(previous_fit)
            guess_run.update(guess)
            
            logging.debug("guess_run: " + str(guess_run))
            logging.debug("extra_args_run: " + str(extra_args_run))
            logging.debug("self.ll: " + str(self.ll))
            res, llval = bestfit_scipy(self.ll,
                                       guess=guess_run,
                                       minimize_kwargs=minimize_kwargs,
                                       **extra_args_run)
            previous_fit = res
            res.update(extra_args_run)
            res["ll"] = llval
            res["dl"] = -1.
            res["ul"] = -1.
            ress.append(res)
            logging.debug("ress: " + str(ress))
            
            # All following is just for debugging (basically what blueice does)
            ret = 0.
            for i, (ll, parameter_names, ll_weight) in enumerate(
                    zip(self.lls,
                        self.ll.likelihood_parameters,
                        self.ll.likelihood_weights)):
                pass_kwargs = {k: v
                            for k, v in res.items()
                            if k in parameter_names}
                
                ret += ll_weight * ll(compute_pdf=False,
                                    livetime_days=None,
                                    **pass_kwargs)
                logging.debug(f"LogLikelihoodSum after adding ll({i}): {ret}")

        if compute_confidence_interval:
            ci_guess = deepcopy(ress[-1])
            ci_guess.pop("signal_rate_multiplier", None)
            ci_guess.update({"signal_rate_multiplier": 0})
            ci_args = {
                "llval_best": ress[-1]["ll"],
                "extra_args": extra_args_runs[-1],
                "guess": ci_guess,
            }
            ci_args.update(confidence_interval_args)
            logging.debug("ci_args: " + str(ci_args))
            dl, ul = self.confidence_interval(**ci_args)
            logging.debug("dl: " + str(dl))
            logging.debug("ul: " + str(ul))
            
            ress[-1]["dl"] = dl
            ress[-1]["ul"] = ul

        return ress

    def get_mus(self,
                evaluate_per_likelihood=False,
                livetime_days_multiplier=None,
                **res):
        """Return dictionary of expectation values for all sources
        evaluate_per_likelihood (bool): returns a list of dictionarys with expectations per likelihood"""

        if not (isinstance(livetime_days_multiplier, list)
                or livetime_days_multiplier is None):
            raise Exception(
                "livetime_days_multiplier needs to be list or None")

        if livetime_days_multiplier is not None and len(
                livetime_days_multiplier) != len(self.lls):
            raise Exception(
                "You cannot set the livetime_days_multiplier for only a subset of your sub-likelihoods"
            )

        to_return = []
        for idx, ll in enumerate(self.lls):
            this_res = deepcopy(res)
            ret = {}

            # need to take care that multiplier is not evaluted for sub-likelihood
            parameters = list(ll.rate_parameters.keys()) + list(
                ll.shape_parameters.keys())
            for res_key in res.keys():
                parameter_name = res_key.replace("_rate_multiplier", "")
                parameter_name = parameter_name.replace(
                    "_shape_multiplier", "")
                if parameter_name not in parameters:
                    del this_res[res_key]

            if isinstance(livetime_days_multiplier, list):
                default_livetime = ll.pdf_base_config["livetime_days"]
                this_res.update({
                    "livetime_days":
                    float(livetime_days_multiplier[idx]) *
                    float(default_livetime)
                })

            mus = ll(full_output=True, **this_res)[1]
            for n, mu in zip(ll.source_name_list, mus):
                ret[n] = ret.get(n, 0) + mu
            to_return.append(ret)

        if evaluate_per_likelihood:
            return to_return
        else:
            ret = {}
            for item in to_return:
                for n, mu in item.items():
                    ret[n] = ret.get(n, 0) + mu
            return ret

    def get_parameter_list(self):
        """Return string of rate and shape parameters"""
        ret = [
            n + "_rate_multiplier"
            for n in list(self.ll.rate_parameters.keys())
        ]
        ret += list(self.ll.shape_parameters.keys())
        ret += ["ll", "dl", "ul"]
        return ret

    def write_toydata(self):
        toydata_to_file(self.toydata_file,
                        datasets_array=self.datasets_array,
                        dataset_names=self.dataset_names,
                        overwrite_existing_file=True)

    def promote_generate_args(self):
        """Promote generate_args to members of the inferenceObject"""
        for key, value in self.generate_args.items():
            print("promoting", key, "to", value)
            setattr(self, key, value)

    def get_pdfs(self, fit_result, slice_args=None, expected_events=False):
        """
        fit_result (dict): dictionary of fit result
        slice_args (list): list of arguments to slice the pdfs
        """
        if slice_args is None:
            slice_args = [{} for _ in self.lls]
        else:
            if len(slice_args) != len(self.lls):
                raise Exception(
                    "You need to provide a slice_args for each sub-likelihood"
                )

        l_pdfs = []
        for plotter, slice_arg in zip(self.pdf_plotters, slice_args):
            myresult = utils._divide_fit_result(
                plotter.ll, fit_result)
            mypdf = plotter.get_pdf(**myresult, slice_args=slice_arg, expected_events=expected_events)
            l_pdfs.append(mypdf)
        return l_pdfs
