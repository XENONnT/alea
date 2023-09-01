from copy import deepcopy
from unittest import TestCase

from alea.utils import load_yaml
from alea.parameters import Parameters


class TestParameters(TestCase):
    """Test of the Parameters class."""

    @classmethod
    def setUp(cls):
        """Initialise the Parameters instance."""
        cls.config = load_yaml("unbinned_wimp_statistical_model.yaml")
        cls.parameters = Parameters.from_config(cls.config["parameter_definition"])

    def test_from_list(self):
        """Test of the from_list method."""
        only_name_parameters = Parameters.from_list(self.config["parameter_definition"].keys())
        # it is false because only names are assigned
        self.assertFalse(only_name_parameters == self.parameters)

    def test___repr__(self):
        """Test of the __repr__ method."""
        for p in self.parameters:
            if not isinstance(repr(p), str):
                raise ValueError("The __repr__ method does not return the correct string.")
        if not isinstance(repr(self.parameters), str):
            raise TypeError("The __repr__ method does not return a string.")

    def test_deep_copyable(self):
        """Test of whether Parameters instance can be deepcopied."""
        if deepcopy(self.parameters) != self.parameters:
            raise ValueError("Parameters instance cannot be correctly deepcopied.")
