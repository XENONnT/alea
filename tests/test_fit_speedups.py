"""Tests for analytic likelihood gradients, fit start values and warm-started profile fits."""

from unittest import TestCase

import numpy as np
from scipy import stats
import pytest

from alea.models import BlueiceExtendedModel
from alea.model import StatisticalModel
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


@pytest.mark.usefixtures("rm_cache")
class TestUnbinnedGradient(TestCase):
    """Analytic gradient of an unbinned model and its use in fits."""

    def test_unbinned_gradient_and_fit(self):
        model = BlueiceExtendedModel.from_config("unbinned_wimp_statistical_model_simple.yaml")
        self.assertTrue(model.supports_gradient())
        np.random.seed(5)
        model.data = model.generate_data(wimp_rate_multiplier=1.0)
        values = model.parameters()
        values.update(wimp_rate_multiplier=1.1, er_rate_multiplier=1.1)
        values["er_band_shift"] = 0.37  # between anchors (linear interpolation: kinks at anchors)
        ll, grad = model.ll_and_grad(**values)
        self.assertAlmostEqual(ll, model.ll(**values), places=8)
        for name in model.parameters.fittable:
            # larger step: rounding noise of this likelihood dominates for steps below ~1e-5
            numeric = numeric_gradient(model, values, name, h=1e-3)
            self.assertAlmostEqual(grad[name], numeric, delta=1e-4 * max(1.0, abs(numeric)))
        fits = {}
        for use_gradient in (False, True):
            fits[use_gradient] = model.fit(fit_strategy={"use_gradient": use_gradient})[1]
            self.assertTrue(model.minuit_object.valid)
        self.assertAlmostEqual(fits[True], fits[False], delta=1e-3)


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
        self.assertEqual(
            best_fit_warm["wimp_mass"], self.model.parameters["wimp_mass"].nominal_value
        )

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


@pytest.mark.usefixtures("rm_cache")
class TestGradientCompatibility(TestCase):
    """The analytic gradient must not be used where it would not match the likelihood."""

    def test_subclass_overriding_ll(self):
        class ModifiedModel(BlueiceExtendedModel):
            def _ll(self, **generate_values):
                return super()._ll(**generate_values) - 0.5 * generate_values["efficiency_constant"]

        model = ModifiedModel.from_config("binned_ces.yaml")
        self.assertFalse(model.supports_gradient())
        np.random.seed(0)
        model.data = model.generate_data()
        model.fit()
        self.assertEqual(model.minuit_object.ngrad, 0)

    def test_instance_patched_ll(self):
        model = BlueiceExtendedModel.from_config("binned_ces.yaml")
        self.assertTrue(model.supports_gradient())
        original_ll = model.ll
        model.ll = lambda **kwargs: original_ll(**kwargs)
        self.assertFalse(model.supports_gradient())

    def test_constraint_functions_modified_in_place(self):
        model = BlueiceExtendedModel.from_config("binned_ces.yaml")
        ancillary = model._likelihood.likelihood_list[-1]
        name = ancillary._gauss_names[0]
        values = model.parameters()
        ancillary.constraint_functions[name] = stats.norm(values[name] + 1.0, 2.0)
        expected = sum(t(values[n]) for n, t in ancillary.constraint_terms.items())
        self.assertAlmostEqual(ancillary.ancillary_sum(values), expected, places=10)
        _, grad = ancillary.loglikelihood_and_gradient(**values)
        self.assertAlmostEqual(grad[name], 1.0 / 4.0, places=10)


@pytest.mark.usefixtures("rm_cache")
class TestSourceWiseBinnedInterpolation(TestCase):
    """Source-wise shape interpolation of binned likelihoods, linear and C1 (cubic Hermite)."""

    @staticmethod
    def build(source_wise, morpher=None):
        config = load_yaml("binned_ces.yaml")
        term = config["likelihood_config"]["likelihood_terms"][0]
        term["source_wise_interpolation"] = source_wise
        if morpher is not None:
            term["likelihood_config"] = {"morpher": morpher}
        model = BlueiceExtendedModel(**config)
        np.random.seed(0)
        model.data = model.generate_data()
        return model

    @staticmethod
    def random_values(model, rng):
        """Shape parameters anywhere within their anchors, other fittable parameters within 20%."""
        blueice_ll = model._likelihood.likelihood_list[0]
        values = model.parameters()
        for name in model.parameters.fittable:
            if name in blueice_ll.shape_parameters:
                values[name] = rng.uniform(*blueice_ll.get_bounds(name))
            else:
                values[name] *= rng.uniform(0.8, 1.2)
        return values

    def test_source_wise_linear_matches_joint_grid(self):
        joint = self.build(source_wise=False)
        source_wise = self.build(source_wise=True)
        self.assertTrue(source_wise.supports_gradient())
        source_wise.data = joint.data
        rng = np.random.default_rng(3)
        for _ in range(5):
            values = self.random_values(joint, rng)
            self.assertAlmostEqual(source_wise.ll(**values), joint.ll(**values), places=7)

    def test_cubic_hermite_gradient_and_fit(self):
        model = self.build(source_wise=True, morpher="CubicHermiteInterpolator")
        self.assertTrue(model.supports_gradient())
        rng = np.random.default_rng(4)
        for _ in range(3):
            values = self.random_values(model, rng)
            ll, grad = model.ll_and_grad(**values)
            self.assertAlmostEqual(ll, model.ll(**values), places=8)
            for name in model.parameters.fittable:
                numeric = numeric_gradient(model, values, name)
                self.assertAlmostEqual(grad[name], numeric, delta=1e-4 * max(1.0, abs(numeric)))
        fits = {}
        for use_gradient in (False, True):
            fits[use_gradient] = model.fit(fit_strategy={"use_gradient": use_gradient})[1]
            self.assertTrue(model.minuit_object.valid)
        self.assertAlmostEqual(fits[True], fits[False], delta=1e-3)


class TestFitWithExtraStarts(TestCase):
    """A model with two minima: the default start finds the worse one, the extra start the
    better."""

    class TwoWells(StatisticalModel):
        def __init__(self, **kwargs):
            parameter_definition = {
                "x": {
                    "nominal_value": -1.0,
                    "fittable": True,
                    "fit_limits": [-3, 3],
                    "fit_guess": -1.0,
                }
            }
            super().__init__(parameter_definition=parameter_definition, **kwargs)

        def _ll(self, x):
            # maximum -1 at x=-1, a higher maximum at x=+1.5 (a deeper well)
            return -min((x + 1.0) ** 2 + 1.0, 3.0 * (x - 1.5) ** 2)

        def _generate_data(self, x):
            return {}

        def get_expectation_values(self, **kwargs):
            return {}

    def setUp(self):
        self.model = self.TwoWells(fit_strategy={"use_gradient": False})
        self.model.data = {}

    def test_default_start_misses_better_minimum(self):
        best_fit, ll = self.model.fit()
        self.assertAlmostEqual(best_fit["x"], -1.0, places=3)

    def test_extra_start_finds_better_minimum(self):
        best_fit, ll = self.model.fit_with_extra_starts([{"x": 1.4}])
        self.assertAlmostEqual(best_fit["x"], 1.5, delta=0.02)
        self.assertGreater(ll, -1.0 + 1e-6)
        self.assertAlmostEqual(self.model.minuit_object.values["x"], best_fit["x"])

    def test_extra_start_is_clipped_to_fit_limits(self):
        best_fit, ll = self.model.fit_with_extra_starts([{"x": 10.0}])
        self.assertLessEqual(best_fit["x"], 3.0)

    def test_no_extra_starts_equals_fit(self):
        self.assertEqual(self.model.fit_with_extra_starts([])[1], self.model.fit()[1])
