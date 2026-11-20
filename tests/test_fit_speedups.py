"""Tests for analytic likelihood gradients, fit start values and warm-started profile fits."""

from unittest import TestCase

import numpy as np
import pytest

from alea.models import BlueiceExtendedModel
from alea.utils import load_yaml


def numeric_gradient(model, values, name, h=1e-6):
    up, down = dict(values), dict(values)
    step = h * max(1.0, abs(values[name]))
    up[name] += step
    down[name] -= step
    return (model.ll(**up) - model.ll(**down)) / (2 * step)


@pytest.mark.usefixtures("rm_cache")
class TestGradient(TestCase):
    """Analytic gradient of a binned model (binned_ces.yaml) and its use in fits."""

    @classmethod
    def setUpClass(cls):
        cls.model = BlueiceExtendedModel.from_config("binned_ces.yaml")
        np.random.seed(0)
        cls.model.data = cls.model.generate_data()

    def test_supports_gradient(self):
        self.assertTrue(self.model.supports_gradient())
        unbinned = BlueiceExtendedModel.from_config("unbinned_wimp_statistical_model_simple.yaml")
        self.assertFalse(unbinned.supports_gradient())

    def test_ll_and_grad(self):
        values = self.model.parameters()
        values.update(
            xe133_rate_multiplier=1100.0,
            test_flat_rate_multiplier=900.0,
            efficiency_constant=0.83,
        )
        ll, grad = self.model.ll_and_grad(**values)
        self.assertAlmostEqual(ll, self.model.ll(**values), places=8)
        for name in self.model.parameters.fittable:
            numeric = numeric_gradient(self.model, values, name)
            self.assertAlmostEqual(grad[name], numeric, delta=1e-4 * max(1.0, abs(numeric)))

    def test_fit_with_and_without_gradient(self):
        fits = {}
        for use_gradient in (False, True):
            best_fit, ll = self.model.fit(fit_strategy={"use_gradient": use_gradient})
            fits[use_gradient] = (best_fit, ll, self.model.minuit_object)
            self.assertTrue(self.model.minuit_object.valid)
        self.assertGreater(fits[True][2].ngrad, 0)
        self.assertEqual(fits[False][2].ngrad, 0)
        self.assertAlmostEqual(fits[True][1], fits[False][1], delta=1e-3)
        for name in self.model.parameters.fittable:
            scale = max(1.0, abs(fits[False][0][name]))
            self.assertAlmostEqual(
                fits[True][0][name] / scale, fits[False][0][name] / scale, delta=1e-2
            )


class TestAncillaryGradient(TestCase):
    """Gradient of the ancillary term, Gaussian and gamma (sideband) constraints."""

    def test_ancillary_gradient(self):
        model = BlueiceExtendedModel.from_config(
            "unbinned_wimp_statistical_model_sideband_constraint.yaml"
        )
        np.random.seed(1)
        model.data = model.generate_data()
        ancillary = model._likelihood.likelihood_list[-1]
        self.assertTrue(ancillary._other_constraint_terms)  # has a gamma constraint
        values = model.parameters()
        for name in ancillary.constraint_functions:
            values[name] *= 1.1
        value, grad = ancillary.loglikelihood_and_gradient(**values)
        self.assertAlmostEqual(value, ancillary(**values), places=10)
        h = 1e-6
        for name in ancillary.constraint_functions:
            up, down = dict(values), dict(values)
            up[name] += h
            down[name] -= h
            numeric = (ancillary(**up) - ancillary(**down)) / (2 * h)
            self.assertAlmostEqual(grad[name], numeric, delta=1e-5 * max(1.0, abs(numeric)))


@pytest.mark.usefixtures("rm_cache")
class TestFitStartValuesAndWarmStart(TestCase):
    """initial_values / initial_errors in fit, and warm_start_profile in confidence_interval."""

    @classmethod
    def setUpClass(cls):
        cls.model = BlueiceExtendedModel.from_config("unbinned_wimp_statistical_model_simple.yaml")
        np.random.seed(2)
        cls.model.data = cls.model.generate_data(wimp_rate_multiplier=1.0)

    def test_initial_values_and_errors(self):
        best_fit, ll = self.model.fit()
        n_cold = self.model.minuit_object.nfcn
        errors = dict(zip(self.model.minuit_object.parameters, self.model.minuit_object.errors))
        # start values for fixed / non-fittable parameters are ignored
        start = dict(best_fit, wimp_mass=1e6)
        best_fit_warm, ll_warm = self.model.fit(initial_values=start, initial_errors=errors)
        self.assertLess(self.model.minuit_object.nfcn, n_cold)
        self.assertAlmostEqual(ll_warm, ll, delta=1e-4)
        self.assertEqual(best_fit_warm["wimp_mass"], self.model.parameters["wimp_mass"].nominal_value)

    def test_warm_start_profile(self):
        intervals = {}
        for warm_start in (False, True):
            intervals[warm_start] = self.model.confidence_interval(
                "wimp_rate_multiplier",
                confidence_interval_kind="central",
                fit_strategy={"warm_start_profile": warm_start},
            )
        for cold, warm in zip(intervals[False], intervals[True]):
            self.assertAlmostEqual(warm, cold, delta=1e-3 * max(1.0, abs(cold)))

    def test_default_fit_strategy_not_shared(self):
        config = load_yaml("unbinned_wimp_statistical_model_simple.yaml")
        BlueiceExtendedModel(**config, fit_strategy={"minuit_strategy": 2})
        other = BlueiceExtendedModel(**config)
        self.assertEqual(other._fit_strategy["minuit_strategy"], 1)
