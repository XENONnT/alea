from unittest import TestCase

from alea.models import BlueiceExtendedModel
from alea.utils import load_yaml


class TestTemplateSource(TestCase):
    """Test of the TemplateSource class."""

    def test_init_templates(self):
        """Test whether we can initialize template sources."""
        parameter_definition = load_yaml("unbinned_wimp_statistical_model.yaml")[
            "parameter_definition"
        ]
        likelihood_config = load_yaml("test_template_source.yaml")["likelihood_config"]
        model = BlueiceExtendedModel(parameter_definition, likelihood_config)
        model.nominal_expectation_values
