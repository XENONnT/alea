from pydoc import locate  # to lookup likelihood class
from alea.statistical_model import StatisticalModel
from alea.simulators import BlueiceDataGenerator
from alea.utils import adapt_likelihood_config_for_blueice
import yaml
import numpy as np
import scipy.stats as stats
from blueice.likelihood import LogAncillaryLikelihood
from blueice.likelihood import LogLikelihoodSum
# from inference_interface import dict_to_structured_array


class BlueiceExtendedModel(StatisticalModel):
    def __init__(self, parameter_definition: dict, likelihood_terms: dict):
        """
        # TODO write docstring
        """
        super().__init__(parameter_definition=parameter_definition)
        self._likelihood = self._build_ll_from_config(likelihood_terms)
        self.likelihood_names = [c["name"] for c in likelihood_terms]
        self.likelihood_names.append("ancillary_likelihood")
        self.data_generators = self._build_data_generators()

        # TODO analysis_space should be inferred from the data (assert that all sources have the same analysis space)

    @classmethod
    def from_config(cls, config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def _ll(self, **generate_values):
        # TODO: Does this make sense?
        return self._likelihood(**generate_values)

    def _generate_data(self, **generate_values):
        # generate_values are already filtered and filled by the nominal values through the generate_data method in the parent class
        science_data = self._generate_science_data(**generate_values)
        ancillary_keys = self.parameters.with_uncertainty.names
        generate_values_anc = {k: v for k, v in generate_values.items() if k in ancillary_keys}
        ancillary_measurements = self._generate_ancillary_measurements(
            **generate_values_anc)
        # generate_values = dict_to_structured_array(generate_values)
        return science_data + [ancillary_measurements] + [generate_values]

    def _generate_science_data(self, **generate_values):
        science_data = [gen.simulate(**generate_values)
                        for gen in self.data_generators]
        return science_data

    def _generate_ancillary_measurements(self, **generate_values):
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
        # TODO: Do we need this as a structured array?
        # ancillary_measurements = dict_to_structured_array(ancillary_measurements)

        return ancillary_measurements

    # TODO: Override uncertainty setter to also set the uncertainty of the ancillary ll term (func_args). Or for now override the uncertainty setter to not work and raise a warning.

    @property
    def data(self):
        return super().data

    @data.setter
    def data(self, data):
        """
        Overrides default setter. Will also set the data of the blueice ll.
        Data-sets are expected to be in the form of a list of one or more structured arrays-- representing the data-sets of one or more likelihood terms.
        """
        # iterate through all likelihood terms and set the science data in the blueice ll
        # last entry in data are the generate_values
        for d, ll_term in zip(data[:-1], self._likelihood.likelihood_list):
            ll_term.set_data(d)

        self._data = data

    def get_expectation_values(self, **kwargs):
        """
        return total expectation values (summed over all likelihood terms with the same name)
        given a number of named parameters (kwargs)
        """
        ret = dict()
        for ll in self._likelihood.likelihood_list[:-1]: # ancillary likelihood does not contribute
            mus = ll(full_output=True, **kwargs)[1]
            for n, mu in zip(ll.source_name_list, mus):
                ret[n] = ret.get(n, 0) + mu
        return ret

    def _build_ll_from_config(self, likelihood_terms):
        # iterate through ll_config and build blueice ll
        lls = []
        for config in likelihood_terms:
            likelihood_object = locate(config["likelihood_type"])
            blueice_config = adapt_likelihood_config_for_blueice(config)
            blueice_config["livetime_days"] = self.parameters[
                blueice_config["livetime_parameter"]].nominal_value
            ll = likelihood_object(blueice_config)
            # Set rate parameters
            for source in config["sources"]:
                for param_name in source["parameters"]:
                    if self.parameters[param_name].type == "rate":
                        # TODO: Check that only one rate per source is set?
                        if param_name.endswith("_rate_multiplier"):
                            param_name = param_name.replace("_rate_multiplier", "")
                            ll.add_rate_parameter(param_name, log_prior=None)
                        else:
                            NotImplementedError
            # TODO: Set shape parameters

            ll.prepare()
            lls.append(ll)
        # Ancillary likelihood
        ll = CustomAncillaryLikelihood(self.parameters.with_uncertainty)
        lls.append(ll)

        # TODO: Include likelihood_weights
        return LogLikelihoodSum(lls, likelihood_weights=None)

    def _build_data_generators(self):
        # last one is AncillaryLikelihood
        # IDEA: Also implement data generator for ancillary ll term.
        return [BlueiceDataGenerator(ll_term) for ll_term in self._likelihood.likelihood_list[:-1]]

# Build wrapper to conveniently define a constraint likelihood term


class CustomAncillaryLikelihood(LogAncillaryLikelihood):
# TODO: Make sure the functions and terms are properly implemented now.
    def __init__(self, parameters):
        self.parameters = parameters
        # check that there are no None values in the uncertainties dict
        assert set(self.parameters.uncertainties.keys()) == set(self.parameters.names)
        parameter_list = self.parameters.names

        self.constraint_functions = self._get_constraint_functions()
        super().__init__(func=self.ancillary_likelihood_sum,
                         parameter_list=parameter_list,
                         config=self.parameters.nominal_values)

    @property
    def constraint_terms(self):
        return {name: func.logpdf for name, func in self.constraint_functions.items()}

    def set_data(self, d: dict):
        # data in this case is a set of ancillary measurements.
        # This results in shifted constraint terms.
        assert set(d.keys()) == set(self.parameters.names)
        self.constraint_functions = self._get_constraint_functions(**d)

    def _get_constraint_functions(self, **generate_values) -> dict:
        central_values = self.parameters(**generate_values)
        constraint_functions = {}
        for name, uncertainty in self.parameters.uncertainties.items():
            param = self.parameters[name]
            if param.relative_uncertainty:
                uncertainty *= param.nominal_value
            if isinstance(uncertainty, float):
                func = stats.norm(central_values[name],
                                  uncertainty)
            else:
                # TODO: Implement str-type uncertainties
                NotImplementedError(
                    "Only float uncertainties are supported at the moment.")
            constraint_functions[name] = func
        return constraint_functions

    def ancillary_likelihood_sum(self, evaluate_at: dict):
        return np.sum([term(evaluate_at[name]) for name, term in self.constraint_terms.items()])
