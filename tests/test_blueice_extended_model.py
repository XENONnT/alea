from os import remove
import pytest
from unittest import TestCase

from blueice.likelihood import LogLikelihoodSum
from alea.utils import load_yaml
from alea.models import BlueiceExtendedModel, CustomAncillaryLikelihood


@pytest.mark.usefixtures("rm_cache")
class TestBlueiceExtendedModel(TestCase):
    """Test of the BlueiceExtendedModel class."""

    @classmethod
    def setUp(cls):
        """Initialise the BlueiceExtendedModel instance"""
        cls.configs = [load_yaml('unbinned_wimp_statistical_model.yaml'),
                       load_yaml('unbinned_wimp_statistical_model_simple.yaml')]
        n = [len(c['likelihood_config']['likelihood_terms']) for c in cls.configs]
        cls.n_likelihood_terms = n
        cls.set_new_models(cls)

    def set_new_models(self):
        """Set a new BlueiceExtendedModel instance"""
        models = []
        for config in self.configs:
            models.append(BlueiceExtendedModel(
                parameter_definition=config['parameter_definition'],
                likelihood_config=config['likelihood_config'],
            ))
        self.models = models

    def get_expectation_values(self):
        # normalization of templates
        nominal_values = {"er": 200, "wimp": 10}
        expectation_values = []
        for config, model in zip(self.configs, self.models):
            this_expectation_dict = {}
            ll_c = config['likelihood_config']
            for ll_term in ll_c['likelihood_terms']:
                livetime_parameter = ll_term['livetime_parameter']
                livetime = model.parameters[livetime_parameter].nominal_value
                for source in ll_term['sources']:
                    name = source['name']
                    this_expectation = nominal_values[name]
                    this_expectation *= source.get('histogram_scale_factor', 1.)
                    this_expectation *= model.parameters[f'{name}_rate_multiplier'].nominal_value
                    this_expectation *= livetime
                    if name in this_expectation_dict:
                        this_expectation_dict[name] += this_expectation
                    else:
                        this_expectation_dict[name] = this_expectation
            expectation_values.append(this_expectation_dict)
        return expectation_values

    def test_expectation_values(self):
        """Test of the expectation_values method"""

        self.set_new_models()
        naive_expectation_values = self.get_expectation_values()
        for model, naive_vals in zip(self.models, naive_expectation_values):
            expectation_values = model.get_expectation_values()

            # should avoid accidentally set data
            is_data_set = False
            for ll_term in model.likelihood_list[:-1]:
                is_data_set |= ll_term.is_data_set
            if is_data_set:
                raise ValueError('Data should not be set after get_expectation_values.')

            # Check whether the expectation values are correct
            for k, v in expectation_values.items():
                self.assertEqual(v, naive_vals[k])

            # Check whether scaling works
            scaling_factor = 2.
            new_expectation_values = model.get_expectation_values(
                wimp_rate_multiplier=scaling_factor,
                er_rate_multiplier=scaling_factor)
            for k, v in expectation_values.items():
                self.assertEqual(v * scaling_factor, new_expectation_values[k])

    def test_generate_data(self):
        """Test of the generate_data method"""
        for model, n in zip(self.models, self.n_likelihood_terms):
            data = model.generate_data()
            toydata_file = 'simple_data.h5'
            model.store_data(
                toydata_file,
                [data])
            remove(toydata_file)
            self.assertEqual(len(data), n + 2)
            if not (('ancillary_likelihood' in data) and ('generate_values' in data)):
                raise ValueError('Data does not contain ancillary_likelihood and generate_values.')
            for k, v in data.items():
                if k in {'ancillary_likelihood', 'generate_values'}:
                    continue
                elif 'source' not in v.dtype.names:
                    raise ValueError('Data does not contain source information.')

    def test_likelihood(self):
        """Test of the _likelihood attribute"""
        for model, n in zip(self.models, self.n_likelihood_terms):
            # Check whether the likelihood is correctly set
            self.assertIsInstance(model._likelihood, LogLikelihoodSum)
            self.assertIsInstance(model.likelihood_list[-1], CustomAncillaryLikelihood)

            # Check length of likelihood
            self.assertEqual(
                len(model.likelihood_list), n + 1)

            # Check whether the likelihood is callable
            model.data = model.generate_data()
            model.ll()

    def test_fit(self):
        """Test of the fit method"""
        for model in self.models:
            model.data = model.generate_data()
            fit_result, max_llh = model.fit()

            # check whether all parameters are in fit_result
            self.assertEqual(set(model.parameters.names),
                             set(fit_result.keys()))

            # check that non-fittable parameters are not fitted
            for p in model.parameters:
                if not p.fittable:
                    self.assertEqual(p.nominal_value, fit_result[p.name])

            # check that values are in fit limits
            for p in model.parameters:
                p.value_in_fit_limits(fit_result[p.name])

            # check that likelihood is maximized
            self.assertEqual(max_llh, model.ll(**fit_result))

            # check that fixing all parameters to nominal works
            fit_result_fixed, _ = model.fit(**model.parameters())
            for p in model.parameters:
                self.assertEqual(p.nominal_value, fit_result_fixed[p.name])


class TestCustomAncillaryLikelihood(TestCase):
    """Test of the CustomAncillaryLikelihood class."""

    def test_ancillary_likelihood(self):
        """Test of the ancillary_likelihood method."""
        # TODO:
        pass
