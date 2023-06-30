from alea.statistical_model import StatisticalModel
from alea.simulators import BlueiceDataGenerator
import yaml
import numpy as np
import scipy.stats as stats
from blueice.likelihood import LogAncillaryLikelihood


class BlueiceExtendedModel(StatisticalModel):
    def __init__(self, parameter_definition: dict, ll_config: dict):
        """
        # TODO write docstring
        """
        super().__init__(parameter_definition=parameter_definition)
        self.parameters_of_ll_terms = self._get_parameters_of_ll_terms(
            ll_config)
        self._ll = self._build_ll_from_config(ll_config)
        self.likelihood_names = [c["name"]
                                 for c in ll_config["likelihood_terms"]]
        self.likelihood_names.append("ancillary_likelihood")
        self.data_generators = self._build_data_generators()

        # TODO analysis_space should be inferred from the data (assert that all sources have the same analysis space)

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    # def _ll(self, **kwargs):
    #     # TODO
    #     # IDEA Set data to blueice ll (or maybe better to set it before explicitly
    #     # since in the fit this will be called frequently but the data won't change.)
    #     # IDEA Maybe one could then define self._ll directly in the init instead of _ll_blueice?

    #     pass

    def _generate_data(self, **generate_values):
        # generate_values are already filtered and filled by the nominal values through the generate_data method in the parent class
        science_data = self._generate_science_data(**generate_values)
        ancillary_measurements = self._generate_ancillary_measurements(
            **generate_values)
        # TODO Check dtype of each dataset
        return science_data + [ancillary_measurements] + [generate_values]

    def _generate_science_data(self, **generate_values):
        science_data = [gen.simulate(**generate_values)
                        for gen in self.data_generators]
        return science_data

    def _generate_ancillary_measurements(self, **generate_values):
        ancillary_measurements = {}
        ancillary_generators = self._ll[-1]._get_constraint_functions(**generate_values)
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
        for d, ll_term in zip(data["science_data"], self.ll.likelihood_list):
            ll_term.set_data(d)
            # TODO: Convert to str-arr
        # TODO: implemment the set_data also for the ancillary measurement likelihood term
        # TODO Frankenstein our own Likelihood term (wrapper class over blueice)

        # generate_values = data["generate_values"]
        # anc_meas = data["ancillary_measurements"]
        # TODO: Set ancillary measurements for rate parameters
        # TODO: Make sure to only set each parameter once (maybe like constraint_already_set)
        # TODO: Set ancillary measurements for shape parameters
        # TODO: Define both here and in the ll that the constraint is put only in the first term that contains the parameter
        # TODO: use .in_likelihood_term() method for parameters?

        self._data = data

    @property
    def nominal_expectation_values(self):
        # TODO
        # IDEA also enable a setter that changes the rate parameters?
        pass

    def get_expectation_values(self, **kwargs):
        # TODO
        pass

    def _build_ll_from_config(self, ll_config):
        # TODO iterate through ll_config and build blueice ll
        # IDEA maybe add a dict with the ll names as keys and the corresponding blueice ll terms as values?
        # IDEA Maybe simply return ll in the end and spare the def of _ll?
        # IDEA Or better define a _ll_blueice and call this in _ll to make it more readable?
        ll = None
        return ll

    def _add_rate_parameters(self):
        # TODO
        # TODO: Check if already set
        pass

    def _add_shape_parameters(self):
        # TODO
        # TODO: Check if already set
        pass

    def _build_data_generators(self):
        # last one is AncillaryLikelihood
        # IDEA: Also implement data generator for ancillary ll term.
        return [BlueiceDataGenerator(ll_term) for ll_term in self.ll.likelihood_list[:-1]]

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
