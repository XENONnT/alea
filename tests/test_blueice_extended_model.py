from os import remove
import pytest
from unittest import TestCase
from copy import deepcopy
import numpy as np

from inference_interface import toydata_from_file

from blueice.likelihood import LogLikelihoodSum
from alea.utils import load_yaml
from alea.models import BlueiceExtendedModel, CustomAncillaryLikelihood


@pytest.mark.usefixtures("rm_cache")
class TestBlueiceExtendedModel(TestCase):
    """Test of the BlueiceExtendedModel class."""

    @classmethod
    def setUp(cls):
        """Initialise the BlueiceExtendedModel instance."""
        cls.configs = [
            load_yaml("unbinned_wimp_statistical_model.yaml"),
            load_yaml("unbinned_wimp_statistical_model_simple.yaml"),
            load_yaml("unbinned_wimp_statistical_model_index_fitting.yaml"),
        ]
        ns = [len(c["likelihood_config"]["likelihood_terms"]) for c in cls.configs]
        cls.n_likelihood_terms = ns
        cls.toydata_filename = "simple_data.ii.h5"
        cls.set_new_models(cls)

    def set_new_models(self):
        """Set a new BlueiceExtendedModel instance."""
        models = []
        for config in self.configs:
            models.append(
                BlueiceExtendedModel(
                    parameter_definition=config["parameter_definition"],
                    likelihood_config=config["likelihood_config"],
                )
            )
        self.models = models

    def test_deep_copyable(self):
        """Test of whether BlueiceExtendedModel instance can be deepcopied."""
        for model in self.models:
            try:
                model.data = model.generate_data()
                deepcopy(model)
            except Exception:
                raise ValueError("BlueiceExtendedModel instance cannot be correctly deepcopied.")

    def get_expectation_values(self):
        # normalization of templates
        nominal_values = {"er": 200, "wimp": 10}
        expectation_values = []
        for config, model in zip(self.configs, self.models):
            this_expectation_dict = {}
            ll_c = config["likelihood_config"]
            for ll_term in ll_c["likelihood_terms"]:
                livetime_parameter = ll_term["livetime_parameter"]
                livetime = model.parameters[livetime_parameter].nominal_value
                for source in ll_term["sources"]:
                    name = source["name"]
                    this_expectation = nominal_values[name]
                    this_expectation *= source.get("histogram_scale_factor", 1.0)
                    this_expectation *= model.parameters[f"{name}_rate_multiplier"].nominal_value
                    this_expectation *= livetime
                    if name in this_expectation_dict:
                        this_expectation_dict[name] += this_expectation
                    else:
                        this_expectation_dict[name] = this_expectation
            expectation_values.append(this_expectation_dict)
        return expectation_values

    def test_all_source_names(self):
        """Test of the all_source_names method."""
        for config, model in zip(self.configs, self.models):
            _source_names = set()
            for ll_t in config["likelihood_config"]["likelihood_terms"]:
                _source_names.update([s["name"] for s in ll_t["sources"]])
            source_names = model.all_source_names
            self.assertEqual(source_names, sorted(_source_names))

    def test_expectation_values(self):
        """Test of the expectation_values method."""

        self.set_new_models()
        naive_expectation_values = self.get_expectation_values()
        for model, naive_vals in zip(self.models, naive_expectation_values):
            expectation_values = model.get_expectation_values()

            # should avoid accidentally set data
            is_data_set = False
            for ll_term in model.likelihood_list[:-1]:
                is_data_set |= ll_term.is_data_set
            if is_data_set:
                raise ValueError("Data should not be set after get_expectation_values.")

            # Check whether the expectation values are correct
            for k, v in expectation_values.items():
                self.assertEqual(v, naive_vals[k])

            # Check whether scaling works
            scaling_factor = 2.0
            new_expectation_values = model.get_expectation_values(
                wimp_rate_multiplier=scaling_factor, er_rate_multiplier=scaling_factor
            )
            for k, v in expectation_values.items():
                self.assertEqual(v * scaling_factor, new_expectation_values[k])

    def test_get_expectation_values_per_likelihood_term(self):
        """Test of the get_expectation_values method with per_likelihood=True."""
        self.set_new_models()
        for model in self.models:
            vals_per = model.get_expectation_values(per_likelihood_term=True)
            vals_total = model.get_expectation_values(per_likelihood_term=False)

            # Manually sum up the per_likelihood_term expectation values
            summed_vals = {}
            for term in vals_per.values():
                for key, value in term.items():
                    summed_vals[key] = summed_vals.get(key, 0) + value

            # Check whether the summed values are equal to the total values
            for key, summed_val in summed_vals.items():
                self.assertEqual(summed_val, vals_total[key])

    def test_store_data(self):
        """Test of the generate_data method."""
        for model, n in zip(self.models, self.n_likelihood_terms):
            data = model.generate_data()
            model.data = data
            data_list_of_dict = [data]
            data_list_of_ordereddict = [model.data]
            data_list_of_list = [[data[k] for k in data.keys()]]
            for d in [data_list_of_dict, data_list_of_ordereddict, data_list_of_list]:
                model.store_data(self.toydata_filename, d)
                remove(self.toydata_filename)

    def test_generate_data(self):
        """Test of the generate_data method."""
        for model, n in zip(self.models, self.n_likelihood_terms):
            data = model.generate_data()
            self.assertEqual(len(data), n + 2)
            if not (("ancillary" in data) and ("generate_values" in data)):
                raise ValueError("Data does not contain ancillary and generate_values.")
            for k, v in data.items():
                if k in {"ancillary", "generate_values"}:
                    continue
                elif "source" not in v.dtype.names:
                    raise ValueError("Data does not contain source information.")
            with self.assertRaises(
                TypeError, msg="Should raise error when directly instantiating StatisticalModel"
            ):
                model.data = data
                model.data["ancillary"] = None

    def test_likelihood(self):
        """Test of the _likelihood attribute."""
        for model, n in zip(self.models, self.n_likelihood_terms):
            # Check whether the likelihood is correctly set
            self.assertIsInstance(model._likelihood, LogLikelihoodSum)
            self.assertIsInstance(model.likelihood_list[-1], CustomAncillaryLikelihood)

            # Check length of likelihood
            self.assertEqual(len(model.likelihood_list), n + 1)

            # Check whether the likelihood is callable
            model.data = model.generate_data()
            model.ll()

    def test_fit(self):
        """Test of the fit method."""
        for model in self.models:
            model.data = model.generate_data()
            fit_result, max_llh = model.fit()

            # check whether all parameters are in fit_result
            self.assertEqual(set(model.parameters.names), set(fit_result.keys()))

            # check that non-fittable parameters are not fitted
            for p in model.parameters:
                if not p.fittable:
                    self.assertEqual(p.nominal_value, fit_result[p.name])

            # check that values are in fit limits
            for p in model.parameters:
                p.value_in_fit_limits(fit_result[p.name])

            # check that likelihood is maximized
            np.testing.assert_almost_equal(max_llh, model.ll(**fit_result), decimal=2)

            # check that fixing all parameters to nominal works
            fit_result_fixed, _ = model.fit(**model.parameters())
            for p in model.parameters:
                self.assertEqual(p.nominal_value, fit_result_fixed[p.name])

    def test_store_real_data(self):
        """Test of the store_real_data method."""
        for model, n in zip(self.models, self.n_likelihood_terms):
            data = model.generate_data()
            model.store_real_data(self.toydata_filename, list(data.values())[:n])
            toydata, _ = toydata_from_file(self.toydata_filename)
            model.data = toydata[0]
            model.fit()
            remove(self.toydata_filename)

    def test_needs_reinit(self):
        """Test of the needs_reinit property of parameters."""
        for model in self.models:
            self.assertTrue(model.parameters["wimp_mass"].needs_reinit)
            # check that a ValueError is raised when trying to set the nominal parameter
            with self.assertRaises(ValueError):
                model.parameters["wimp_mass"].nominal_value = 1
            # check that a ValueError is raised when calling the parameters with another value
            with self.assertRaises(ValueError):
                model.parameters(wimp_mass=1)
            with self.assertRaises(ValueError):
                model.get_expectation_values(wimp_mass=1)
            model.data = model.generate_data()
            with self.assertRaises(ValueError):
                model.fit(wimp_mass=1)

    def test_get_source_histograms(self):
        """Test of the get_source_histograms method."""
        for model in self.models:
            mus = model.get_expectation_values(per_likelihood_term=True)
            for ll_name, ll_term in zip(model.likelihood_names[:-1], model.likelihood_list[:-1]):
                source_histograms = model.get_source_histograms(ll_name)
                self.assertEqual(sorted(source_histograms.keys()), sorted(ll_term.source_name_list))

                # check whether the correct source histograms are returned
                for s_name, histogram in source_histograms.items():
                    source_index = ll_term.source_name_list.index(s_name)
                    blueice_source = ll_term.base_model.sources[source_index]
                    blueice_hist = blueice_source._pdf_histogram.histogram
                    np.testing.assert_almost_equal(blueice_hist, histogram, decimal=10)

                # check that expected_events boolean works
                source_histograms = model.get_source_histograms(ll_name, expected_events=True)
                for s_name, histogram in source_histograms.items():
                    mu = mus[ll_name][s_name]
                    sum_hist = histogram.n
                    np.testing.assert_almost_equal(mu, sum_hist, decimal=4)

            # check that model.likelihood_names[-1] fails
            with self.assertRaises(ValueError):
                model.get_source_histograms(model.likelihood_names[-1])

            # check that invalid likelihood names fail
            with self.assertRaises(ValueError):
                model.get_source_histograms("alea_iacta_est")

    def test_sorted_returns(self):
        """Test if sources are sorted in the same way for all return dicts."""
        for model in self.models:
            mus_per_ll = model.get_expectation_values(per_likelihood_term=True)
            mus = model.get_expectation_values()
            hist_per_ll = {}
            for ll_name in model.likelihood_names[:-1]:
                hist_per_ll[ll_name] = model.get_source_histograms(ll_name)
            # check that keys are the same for each SR
            for ll_name in model.likelihood_names[:-1]:
                self.assertEqual(mus_per_ll[ll_name].keys(), hist_per_ll[ll_name].keys())
            # check that global keys are the same
            all_keys = {v for d in mus_per_ll.values() for v in d.keys()}
            all_keys = sorted(all_keys)
            self.assertEqual(all_keys, sorted(mus.keys()))

    def test_apply_efficiency(self):
        """Test if efficiencies are properly applied."""
        model = self.models[0]
        nominal_n = model.get_expectation_values()["wimp"]
        for signal_eff in [0, 0.5, 1.0, 2.0]:
            model.data = model.generate_data()
            n = model.get_expectation_values(signal_efficiency=signal_eff)["wimp"]
            self.assertEqual(n, nominal_n * signal_eff)
