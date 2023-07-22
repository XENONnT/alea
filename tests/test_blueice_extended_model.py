import yaml
import pkg_resources
from unittest import TestCase

from blueice.likelihood import LogLikelihoodSum
from alea.blueice_extended_model import BlueiceExtendedModel, CustomAncillaryLikelihood


class TestBlueiceExtendedModel(TestCase):
    """Test of the BlueiceExtendedModel class"""

    def __init__(self, *args, **kwargs):
        """Initialize the BlueiceExtendedModel class"""
        super().__init__(*args, **kwargs)
        # TODO: put the file path searching in utils.py
        with open(pkg_resources.resource_filename(
            'alea', 'examples/unbinned_wimp_statistical_model.yaml'), 'r') as f:
            self.config = yaml.safe_load(f)
        self.n_likelihood_terms = len(self.config['likelihood_config']['likelihood_terms'])
        self.model = BlueiceExtendedModel(
            parameter_definition=self.config['parameter_definition'],
            likelihood_config=self.config['likelihood_config'],
        )

    def test_expectation_values(self):
        """Test of the expectation_values method"""
        self.model.data = self.model.generate_data()
        expectation_values = self.model.get_expectation_values()
        # TODO: assert expectation values after test template source
        # self.assertEqual()

    def test_generate_data(self):
        """Test of the generate_data method"""
        data = self.model.generate_data()
        self.assertEqual(
            len(data), self.n_likelihood_terms + 2)
        if not all(['source' in d.dtype.names for d in data[:-2]]):
            raise ValueError('Data does not contain source information.')

    def test_likelihood(self):
        """Test of the _likelihood attribute"""
        self.assertIsInstance(self.model._likelihood, LogLikelihoodSum)
        self.assertIsInstance(self.model._likelihood.likelihood_list[-1], CustomAncillaryLikelihood)
        self.assertEqual(
            len(self.model._likelihood.likelihood_list),
            self.n_likelihood_terms + 1)
        self.model.data = self.model.generate_data()
        self.model._likelihood()

    def test_fit(self):
        """Test of the fit method"""
        self.model.data = self.model.generate_data()
        fit_result, max_llh = self.model.fit()
        # TODO: check whether all parameters are in fit_result
        # and whether fittable parameters are fitted


class TestCustomAncillaryLikelihood(TestCase):
    """Test of the CustomAncillaryLikelihood class"""

    def test_ancillary_likelihood(self):
        """Test of the ancillary_likelihood method"""
        pass
