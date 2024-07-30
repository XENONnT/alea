from copy import deepcopy
from unittest import TestCase

from alea.utils import load_yaml
from alea.parameters import Parameters


class TestParameters(TestCase):
    """Test of the Parameters class."""

    def setUp(self):
        """Initialise the Parameters instance."""
        filenames = [
            "unbinned_wimp_statistical_model.yaml",
            "unbinned_wimp_statistical_model_mass_dependent_efficiency.yaml",
        ]
        configs = []
        for fn in filenames:
            configs.append(load_yaml(fn)["parameter_definition"])
        self.configs = configs

        parameters_list = []
        for config in self.configs:
            parameters_list.append(Parameters.from_config(config))
        self.parameters_list = parameters_list

    def test_from_list(self):
        """Test of the from_list method."""
        for config, parameters in zip(self.configs, self.parameters_list):
            only_name_parameters = Parameters.from_list(config.keys())
            # it is false because only names are assigned
            self.assertFalse(only_name_parameters == parameters)

    def test___repr__(self):
        """Test of the __repr__ method."""
        for parameters in self.parameters_list:
            for p in parameters:
                if not isinstance(repr(p), str):
                    raise ValueError("The __repr__ method does not return the correct string.")
            if not isinstance(repr(parameters), str):
                raise TypeError("The __repr__ method does not return a string.")

    def test_deep_copyable(self):
        """Test of whether Parameters instance can be deepcopied."""
        for parameters in self.parameters_list:
            if deepcopy(parameters) != parameters:
                raise ValueError("Parameters instance cannot be correctly deepcopied.")

    def test_conditional_parameter(self):
        """Test of the ConditionalParameter class."""
        config = self.configs[1]
        parameters = self.parameters_list[1]
        nominal_wimp_mass = config["wimp_mass"]["nominal_value"]
        signal_eff_uncert_dict = config["signal_efficiency"]["uncertainty"]

        # Directly accessing the property should return the value
        # under nominal conditions
        val = parameters.signal_efficiency.uncertainty
        expected_val = signal_eff_uncert_dict[nominal_wimp_mass]
        self.assertEqual(val, expected_val)

        # Calling without kwargs should return the value
        # under nominal conditions
        val = parameters.signal_efficiency().uncertainty
        expected_val = signal_eff_uncert_dict[nominal_wimp_mass]
        self.assertEqual(val, expected_val)

        # Calling with kwargs should return the value under
        # the specified conditions
        for wimp_mass, expected_val in signal_eff_uncert_dict.items():
            val = parameters.signal_efficiency(wimp_mass=wimp_mass).uncertainty
            self.assertEqual(val, expected_val)
