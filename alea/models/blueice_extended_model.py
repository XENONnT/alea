from typing import List
from copy import deepcopy
from pydoc import locate

import yaml
import numpy as np
import scipy.stats as stats
from blueice.likelihood import LogAncillaryLikelihood, LogLikelihoodSum
from inference_interface import dict_to_structured_array, structured_array_to_dict

from alea.model import StatisticalModel
from alea.parameters import Parameters
from alea.simulators import BlueiceDataGenerator
from alea.utils import adapt_likelihood_config_for_blueice, get_template_folder_list


class BlueiceExtendedModel(StatisticalModel):
    """
    A statistical model based on blueice likelihoods.

    This class extends the `StatisticalModel` class and provides methods
    for generating data and computing likelihoods based on blueice.

    Args:
        parameter_definition (dict): A dictionary defining the model parameters.
        likelihood_config (dict): A dictionary defining the likelihood.
    """

    def __init__(self, parameter_definition: dict, likelihood_config: dict):
        """Initializes the statistical model.

        Args:
            parameter_definition (dict): A dictionary defining the model parameters.
            likelihood_config (dict): A dictionary defining the likelihood.
        """
        super().__init__(parameter_definition=parameter_definition)
        self._likelihood = self._build_ll_from_config(likelihood_config)
        self.likelihood_names = [t["name"] for t in likelihood_config["likelihood_terms"]]
        self.likelihood_names.append("ancillary_likelihood")
        self.data_generators = self._build_data_generators()

    # TODO analysis_space could be inferred from the data
    # (assert that all sources have the same analysis space)

    @classmethod
    def from_config(cls, config_file_path: str) -> "BlueiceExtendedModel":
        """Initializes the statistical model from a yaml config file.

        Args:
            config_file_path (str): Path to the yaml config file.

        Returns:
            BlueiceExtendedModel: Statistical model.
        """
        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    @property
    def data(self) -> dict:
        """Returns the data of the statistical model."""
        return super().data

    @data.setter
    def data(self, data: dict):
        """
        Overrides default setter. Will also set the data of the blueice ll.
        Data-sets are expected to be in the form of a list of one
        or more structured arrays-- representing the data-sets of one or more likelihood terms.
        """
        # iterate through all likelihood terms and set the science data in the blueice ll
        # last entry in data are the generate_values
        for i, (dataset_name, d) in enumerate(data.items()):
            if dataset_name != "generate_values":
                ll_term = self._likelihood.likelihood_list[i]
                assert dataset_name == ll_term.pdf_base_config["name"], "Likelihood names do not match."
                ll_term.set_data(d)

        self._data = data
        self.is_data_set = True

    def get_expectation_values(self, **kwargs) -> dict:
        """
        Return total expectation values (summed over all likelihood terms with the same name)
        given a number of named parameters (kwargs)
        TODO: Current implementation is not elegant.
        It copied the llh and sets the data to the copied llh,
        because the call of llh needs data to be set.
        But data is not needed for the expectation values.
        We should update this function in the future after we stop using blueice.
        """
        generate_values = self.parameters(**kwargs)  # kwarg or nominal value
        ret = dict()

        # calling ll need data to be set
        self_copy = deepcopy(self)
        self_copy.data = self_copy.generate_data()

        # TODO: Make a self.likelihood_temrs dict with the likelihood names as keys and the corresponding likelihood terms as values.
        # ancillary likelihood does not contribute
        for ll_term, parameter_names in zip(
                self_copy._likelihood.likelihood_list[:-1],
                self_copy._likelihood.likelihood_parameters):
            # WARNING: This silently drops parameters it can't handle!
            call_args = {k: i for k, i in generate_values.items() if k in parameter_names}

            mus = ll_term(full_output=True, **call_args)[1]
            for n, mu in zip(ll_term.source_name_list, mus):
                ret[n] = ret.get(n, 0) + mu
        return ret

    def _build_ll_from_config(self, likelihood_config: dict) -> "LogLikelihoodSum":
        """
        Iterate through all likelihood terms and build blueice ll

        Args:
            likelihood_config (dict): A dictionary defining the likelihood.
        """
        lls = []

        template_folder_list = get_template_folder_list(likelihood_config)

        # Iterate through each likelihood term in the configuration
        for config in likelihood_config["likelihood_terms"]:
            likelihood_object = locate(config["likelihood_type"])

            blueice_config = adapt_likelihood_config_for_blueice(
                config, template_folder_list)
            blueice_config["livetime_days"] = self.parameters[
                blueice_config["livetime_parameter"]].nominal_value
            for p in self.parameters:
                # adding the nominal rate values will screw things up in blueice!
                # So here we're just adding the nominal values of all other parameters
                if p.ptype != "rate":
                    blueice_config[p.name] = blueice_config.get(p.name, p.nominal_value)

            # add all parameters to extra_dont_hash for each source unless it is used:
            for i, source in enumerate(config["sources"]):
                parameters_to_ignore: List[str] = [
                    p.name for p in self.parameters if (
                        p.ptype == "shape") & (p.name not in source["parameters"])]
                # no efficiency affects PDF:
                parameters_to_ignore += [p.name for p in self.parameters if (p.ptype == "efficiency")]
                parameters_to_ignore += source.get("extra_dont_hash_settings", [])

                # ignore all shape parameters known to this model not named specifically in the source:
                blueice_config["sources"][i]["extra_dont_hash_settings"] = parameters_to_ignore

            ll = likelihood_object(blueice_config)

            for source in config["sources"]:
                # Set rate parameters
                rate_parameters = [
                    p for p in source["parameters"] if self.parameters[p].ptype == "rate"]
                if len(rate_parameters) != 1:
                    raise ValueError(
                        f"Source {source['name']} must have exactly one rate parameter.")
                rate_parameter = rate_parameters[0]
                if rate_parameter.endswith("_rate_multiplier"):
                    rate_parameter = rate_parameter.replace("_rate_multiplier", "")
                    ll.add_rate_parameter(rate_parameter, log_prior=None)
                else:
                    raise NotImplementedError(
                        "Only rate multipliers that end on _rate_multiplier"
                        " are currently supported.")

                # Set efficiency parameters
                if source.get("apply_efficiency", False):
                    self._set_efficiency(source, ll)

                # Set shape parameters
                shape_parameters = [
                    p for p in source["parameters"] if self.parameters[p].ptype == "shape"]
                for p in shape_parameters:
                    anchors = self.parameters[p].blueice_anchors
                    if anchors is None:
                        raise ValueError(f"Shape parameter {p} does not have any anchors.")
                    ll.add_shape_parameter(p, anchors=anchors, log_prior=None)

            ll.prepare()
            lls.append(ll)

        # Ancillary likelihood
        ll = CustomAncillaryLikelihood(self.parameters.with_uncertainty)
        lls.append(ll)

        likelihood_weights = likelihood_config.get("likelihood_weights", None)
        return LogLikelihoodSum(lls, likelihood_weights=likelihood_weights)

    def _build_data_generators(self) -> list:
        # last one is AncillaryLikelihood
        # IDEA: Also implement data generator for ancillary ll term.
        return [
            BlueiceDataGenerator(ll_term) for ll_term in self._likelihood.likelihood_list[:-1]]

    def _ll(self, **generate_values) -> float:
        return self._likelihood(**generate_values)

    def _generate_data(self, **generate_values) -> dict:
        # generate_values are already filtered and filled by the nominal values\
        # through the generate_data method in the parent class
        data = self._generate_science_data(**generate_values)
        ancillary_keys = self.parameters.with_uncertainty.names
        generate_values_anc = {k: v for k, v in generate_values.items() if k in ancillary_keys}
        data["ancillary_likelihood"] = self._generate_ancillary_measurements(
            **generate_values_anc)
        data["generate_values"] = dict_to_structured_array(generate_values)
        return data

    def _generate_science_data(self, **generate_values) -> dict:
        science_data = [gen.simulate(**generate_values)
                        for gen in self.data_generators]
        return dict(zip(self.likelihood_names, science_data))

    def _generate_ancillary_measurements(self, **generate_values) -> dict:
        ancillary_measurements = {}
        anc_ll = self._likelihood.likelihood_list[-1]
        ancillary_generators = anc_ll._get_constraint_functions(**generate_values)
        for name, gen in ancillary_generators.items():
            parameter_meas = gen.rvs()
            # correct parameter_meas if out of bounds
            param = self.parameters[name]
            if not param.value_in_fit_limits(parameter_meas):
                if param.fit_limits[0] is not None and parameter_meas < param.fit_limits[0]:
                    parameter_meas = param.fit_limits[0]
                elif param.fit_limits[1] is not None and parameter_meas > param.fit_limits[1]:
                    parameter_meas = param.fit_limits[1]
            ancillary_measurements[name] = parameter_meas

        return dict_to_structured_array(ancillary_measurements)

    def _set_efficiency(self, source, ll):
        if "efficiency_name" not in source:
            raise ValueError(f"Unspecified efficiency_name for source {source['name']:s}")
        efficiency_name = source["efficiency_name"]

        if efficiency_name not in source["parameters"]:
            raise ValueError(
                f"The efficiency_name for source {source['name']:s}"
                " is not in its parameter list")
        efficiency_parameter = self.parameters[efficiency_name]

        if efficiency_parameter.ptype != "efficiency":
            raise ValueError(f"The parameter {efficiency_name:s} must be an efficiency")
        limits = efficiency_parameter.fit_limits

        if limits[0] < 0:
            raise ValueError(
                f"Efficiency parameters including {efficiency_name:s}"
                " must be constrained to be nonnegative")
        if ~np.isfinite(limits[1]):
            raise ValueError(
                f"Efficiency parameters including {efficiency_name:s}"
                " must be constrained to be finite")
        ll.add_shape_parameter(efficiency_name, anchors=(limits[0], limits[1]))


class CustomAncillaryLikelihood(LogAncillaryLikelihood):
    """
    Custom ancillary likelihood that can be used to add constraint terms
    for parameters of the likelihood.

    Args:
        parameters (Parameters): Parameters object containing the
            parameters to be constrained.
    """

    def __init__(self, parameters: Parameters):
        """Initialize the CustomAncillaryLikelihood.

        Args:
            parameters (Parameters): Parameters object containing the
                parameters to be constrained.
        """
        self.parameters = parameters
        # check that there are no None values in the uncertainties dict
        if set(self.parameters.uncertainties.keys()) != set(self.parameters.names):
            raise ValueError(
                "The uncertainties dict must contain all parameters as keys.")
        parameter_list = self.parameters.names

        self.constraint_functions = self._get_constraint_functions()
        super().__init__(func=self.ancillary_likelihood_sum,
                         parameter_list=parameter_list,
                         config=self.parameters.nominal_values)
        self.pdf_base_config["name"] = "ancillary_likelihood"

    @property
    def constraint_terms(self) -> dict:
        """
        Dict of all constraint terms (logpdf of constraint functions)
        of the ancillary likelihood.
        """
        return {name: func.logpdf for name, func in self.constraint_functions.items()}

    def set_data(self, d: np.array):
        """
        Set the data of the ancillary likelihood (ancillary measurements).

        Args:
            d (np.array): Data of ancillary measurements, stored as numpy array
        """
        # This results in shifted constraint terms.
        d_dict = structured_array_to_dict(d)
        if set(d_dict.keys()) != set(self.parameters.names):
            raise ValueError(
                "The data dict must contain all parameters as keys in CustomAncillaryLikelihood.")
        self.constraint_functions = self._get_constraint_functions(**d_dict)

    def ancillary_likelihood_sum(self, evaluate_at: dict) -> float:
        """Return the sum of all constraint terms.

        Args:
            evaluate_at (dict): Values of the ancillary measurements.

        Returns:
            float: Sum of all constraint terms.
        """
        evaluated_constraint_terms = [
            term(evaluate_at[name]) for name, term in self.constraint_terms.items()
        ]
        return np.sum(evaluated_constraint_terms)

    def _get_constraint_functions(self, **generate_values) -> dict:
        central_values = self.parameters(**generate_values)
        constraint_functions = {}
        for name, uncertainty in self.parameters.uncertainties.items():
            param = self.parameters[name]
            if param.relative_uncertainty:
                uncertainty *= param.nominal_value
            if isinstance(uncertainty, float):
                func = stats.norm(
                    central_values[name], uncertainty)
            else:
                # TODO: Implement str-type uncertainties
                NotImplementedError(
                    "Only float uncertainties are supported at the moment.")
            constraint_functions[name] = func
        return constraint_functions
