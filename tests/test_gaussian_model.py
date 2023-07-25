from unittest import TestCase

from os import remove

import inference_interface

import numpy as np

import scipy.stats as sps

from alea.models import GaussianModel


class TestGaussianModel(TestCase):
    """Test of the Parameters class"""

    @classmethod
    def setUp(cls):
        """
        Initialise the GaussianModel
        """
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
        cls.simple_model = GaussianModel(
            parameter_definition=parameter_definition)

    def test_data_generation(self):
        """
        Test of generate_data and fit method of the GaussianModel class
        """
        # test data generation:
        self.simple_model.data = self.simple_model.generate_data(mu=0, sigma=2)

    def test_data_storage(self):
        # test data store+load:
        toydata_file = 'simple_data.hdf5'
        self.simple_model.data = self.simple_model.generate_data(mu=0, sigma=2)
        self.simple_model.store_data(toydata_file, [self.simple_model.data])
        stored_data = inference_interface.toydata_from_file('simple_data.hdf5')
        assert self.simple_model.data == stored_data[0] , "Stored data disagrees with data!"

        remove("simple_data.hdf5")

    def test_fit(self):
        # test fitting:
        self.simple_model.data = self.simple_model.generate_data(mu=0, sigma=2)
        hat_meas = self.simple_model.data[0]["hat_mu"]
        best_fit, lf = self.simple_model.fit(sigma=2)
        hat_fit = best_fit["mu"]
        np.testing.assert_almost_equal(hat_meas, hat_fit), "best-fit does not agree"
        np.testing.assert_almost_equal(lf, sps.norm(hat_fit, 2).logpdf(hat_meas)) , "likelihood function disagrees"