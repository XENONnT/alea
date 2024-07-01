from unittest import TestCase

from alea.models import BlueiceExtendedModel
from alea.utils import load_yaml


class TestTemplateSource(TestCase):
    """Test of the TemplateSource class."""

    def test_init_templates(self):
        """Test whether we can initialize template sources."""
        model_configs = load_yaml("unbinned_wimp_statistical_model_template_source_test.yaml")
        parameter_definition = model_configs["parameter_definition"]
        likelihood_config = model_configs["likelihood_config"]
        model = BlueiceExtendedModel(parameter_definition, likelihood_config)
        model.nominal_expectation_values

    def test_wrong_analysis_space(self):
        """Test whether initializing with a wrong analysis_space raises error."""
        model_configs = load_yaml("unbinned_wimp_statistical_model_template_source_test.yaml")
        parameter_definition = model_configs["parameter_definition"]
        likelihood_config = model_configs["likelihood_config"]
        # Change the analysis space to a wrong one
        space = likelihood_config["likelihood_terms"][0]["analysis_space"]
        # additive mismatch in cs1
        space[0]["cs1"] = "np.linspace(1, 101, 51)"
        space[1]["cs2"] = "np.geomspace(100, 100000, 51)"
        with self.assertRaises(AssertionError):
            _ = BlueiceExtendedModel(parameter_definition, likelihood_config)

        # multiplicative mismatch in cs2
        space[0]["cs1"] = "np.linspace(0, 100, 51)"
        space[1]["cs2"] = "np.geomspace(101, 101000, 51)"
        with self.assertRaises(AssertionError):
            _ = BlueiceExtendedModel(parameter_definition, likelihood_config)

        # If the dimensions are wrong we should get ValueError
        space[0]["cs1"] = "np.linspace(0, 100, 50)"
        space[1]["cs2"] = "np.geomspace(100, 100000, 51)"
        with self.assertRaises(ValueError):
            _ = BlueiceExtendedModel(parameter_definition, likelihood_config)
