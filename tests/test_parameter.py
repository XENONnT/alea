from copy import deepcopy
from unittest import TestCase

from alea.utils import load_yaml
from alea.parameters import Parameters


class TestParameters(TestCase):
    """Test of the Parameters class"""

    def __init__(self, *args, **kwargs):
        """Initialize the BlueiceExtendedModel class"""
        super().__init__(*args, **kwargs)
        self.config = load_yaml('unbinned_wimp_statistical_model.yaml')
        self.parameters = Parameters.from_config(self.config['parameter_definition'])

    def test_deep_copyable(self):
        """Test of whether Parameters instance can be deepcopied"""
        if deepcopy(self.parameters) != self.parameters:
            raise ValueError('Parameters instance cannot be correctly deepcopied.')
