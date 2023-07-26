from os import remove
from unittest import TestCase

import scipy.stats as sps
import inference_interface

from alea.models import GaussianModel


class TestGaussianModel(TestCase):
    """Test of the GaussianModel class"""

    @classmethod
    def setUp(cls):
        """Initialise the GaussianModel instance"""
        parameter_definition = {
            'mu': {
                'fit_guess': 0.,
                'fittable': True,
                'nominal_value': 0.,
            },
            'sigma': {
                'fit_guess': 1.,
                'fit_limits': [
                    0.,
                    None,
                ],
                'fittable': True,
                'nominal_value': 1.,
            },
        }
        cls.model = GaussianModel(
            parameter_definition=parameter_definition)

    def test_data_generation(self):
        """Test generation of data"""
        self.model.data = self.model.generate_data(mu=0, sigma=2)

    def test_data_storage(self):
        """Test storage of data to file and retrieval of data from file"""
        toydata_file = 'simple_data.hdf5'
        self.model.data = self.model.generate_data(mu=0, sigma=2)
        self.model.store_data(toydata_file, [self.model.data])
        stored_data = inference_interface.toydata_from_file(toydata_file)
        assert self.model.data == stored_data[0], 'Stored data disagrees with data!'
        remove(toydata_file)

    def test_fit_result(self):
        """Test fitting of data"""
        self.model.data = self.model.generate_data(mu=0, sigma=2)
        hat_meas = self.model.data[0]['hat_mu'].item()
        best_fit, lf = self.model.fit(sigma=2)
        hat_fit = best_fit['mu']
        self.assertAlmostEqual(hat_meas, hat_fit)
        self.assertAlmostEqual(lf, sps.norm(hat_fit, 2).logpdf(hat_meas))