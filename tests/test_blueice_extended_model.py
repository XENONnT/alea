from os import remove
import pytest
from unittest import TestCase

from blueice.likelihood import LogLikelihoodSum
from alea.utils import load_yaml
from alea.models import BlueiceExtendedModel, CustomAncillaryLikelihood


@pytest.mark.usefixtures('rm_cache')
class TestBlueiceExtendedModel(TestCase):
    """Test of the BlueiceExtendedModel class"""

    @classmethod
    def setUp(cls):
        """Initialise the BlueiceExtendedModel instance"""
        cls.config = load_yaml('unbinned_wimp_statistical_model.yaml')
        cls.n_likelihood_terms = len(cls.config['likelihood_config']['likelihood_terms'])
        cls.set_new_model(cls)

    def set_new_model(self):
        """Set a new BlueiceExtendedModel instance"""
        self.model = BlueiceExtendedModel(
            parameter_definition=self.config['parameter_definition'],
            likelihood_config=self.config['likelihood_config'],
        )

    def test_expectation_values(self):
        """Test of the expectation_values method"""
        self.set_new_model()
        expectation_values = self.model.get_expectation_values()

        # should avoid accidentally set data
        is_data_set = False
        for ll_term in self.model._likelihood.likelihood_list[:-1]:
            is_data_set |= ll_term.is_data_set
        if is_data_set:
            raise ValueError('Data should not be set after get_expectation_values.')
        # TODO: assert expectation values after manually initialize template source

    def test_generate_data(self):
        """Test of the generate_data method"""
        data = self.model.generate_data()
        toydata_file = 'simple_data.h5'
        self.model.store_data(
            toydata_file,
            [data for k in self.model.likelihood_names])
        remove(toydata_file)
        self.assertEqual(
            len(data), self.n_likelihood_terms + 2)
        if not (('ancillary_likelihood' in data) and ('generate_values' in data)):
            raise ValueError('Data does not contain ancillary_likelihood and generate_values.')
        for k, v in data.items():
            if k in {'ancillary_likelihood', 'generate_values'}:
                continue
            elif 'source' not in v.dtype.names:
                raise ValueError('Data does not contain source information.')

    def test_likelihood(self):
        """Test of the _likelihood attribute"""
        self.assertIsInstance(self.model._likelihood, LogLikelihoodSum)
        self.assertIsInstance(self.model._likelihood.likelihood_list[-1], CustomAncillaryLikelihood)
        self.assertEqual(
            len(self.model._likelihood.likelihood_list),
            self.n_likelihood_terms + 1)
        self.model.data = self.model.generate_data()
        self.model.ll()

    def test_fit(self):
        """Test of the fit method"""
        self.model.data = self.model.generate_data()
        fit_result, max_llh = self.model.fit()
        # TODO:
        # check whether all parameters are in fit_result
        # and whether fittable parameters are fitted
        # and whether the results are in boundaries


class TestCustomAncillaryLikelihood(TestCase):
    """Test of the CustomAncillaryLikelihood class"""

    def test_ancillary_likelihood(self):
        """Test of the ancillary_likelihood method"""
        # TODO:
        pass
