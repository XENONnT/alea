from copy import deepcopy
from unittest import TestCase

from alea.utils import load_yaml
from alea.parameters import Parameters


class TestParameters(TestCase):
    """Test of the Parameters class"""

    @classmethod
    def setUp(cls):
        """Initialise the Parameters instance"""
        config = load_yaml('unbinned_wimp_statistical_model.yaml')
        cls.parameters = Parameters.from_config(config['parameter_definition'])

    def test_deep_copyable(self):
        """Test of whether Parameters instance can be deepcopied"""
        if deepcopy(self.parameters) != self.parameters:
            raise ValueError('Parameters instance cannot be correctly deepcopied.')
