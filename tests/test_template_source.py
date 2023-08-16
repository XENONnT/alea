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
