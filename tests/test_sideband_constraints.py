from unittest import TestCase
from alea.parameters import Parameter, Parameters
from alea.utils import load_yaml


class TestSidebandConstraintConstruction(TestCase):
    def test_sideband_constraint_construction(self):
        """A sideband parameter should store from_sideband, n_sideband, and auto-set uncertainty."""
        p = Parameter(name="tau", nominal_value=1.0, from_sideband=True, n_sideband=10)
        self.assertTrue(p.from_sideband)
        self.assertEqual(p.n_sideband, 10)
        self.assertEqual(p.uncertainty, p.n_sideband)

    def test_from_sideband_none_by_default(self):
        """A normal parameter should have from_sideband=None and n_sideband=None."""
        p = Parameter(name="tau", nominal_value=1.0, uncertainty=0.1)
        self.assertIsNone(p.from_sideband)
        self.assertIsNone(p.n_sideband)

    def test_n_sideband_must_be_positive(self):
        """A non-positive n_sideband should raise a ValueError."""
        for invalid in (-1, 0):
            with self.assertRaisesRegex(ValueError, "n_sideband should be a positive integer"):
                Parameter(name="tau", nominal_value=1.0, from_sideband=True, n_sideband=invalid)

    def test_n_sideband_must_be_integer(self):
        """A non-integer n_sideband should raise a ValueError."""
        with self.assertRaisesRegex(ValueError, "n_obs should be an integer"):
            Parameter(name="tau", nominal_value=1.0, from_sideband=True, n_sideband=10.5)

    def test_n_sideband_must_be_none_if_not_from_sideband(self):
        """Setting n_sideband without from_sideband=True should raise a ValueError."""
        with self.assertRaisesRegex(
            ValueError, "n_sideband should only be set when from_sideband is True"
        ):
            Parameter(name="tau", nominal_value=1.0, n_sideband=10)

    def test_no_uncertainty_when_from_sideband(self):
        """Providing an explicit uncertainty alongside from_sideband=True should raise a
        ValueError."""
        with self.assertRaisesRegex(ValueError, "uncertainty should not be provided"):
            Parameter(
                name="tau", nominal_value=1.0, from_sideband=True, n_sideband=10, uncertainty=0.1
            )


class TestSidebandConstraintConfig(TestCase):
    def setUp(self):
        """Set up a Parameters object from a config file that includes sideband constraints."""
        self.params = None
        filename = "unbinned_wimp_statistical_model_sideband_constraint.yaml"
        self.parameter_config = load_yaml(filename)["parameter_definition"]
        try:
            self.params = Parameters.from_config(self.parameter_config)
        except Exception as e:
            self.fail(f"Failed to create Parameters from config: {e}")

    def test_config_loads_successfully(self):
        """The config file should load without errors and produce a Parameters object."""
        self.params = Parameters.from_config(self.parameter_config)
        self.assertIsNotNone(self.params)

    def test_sideband_set_type(self):
        """The from_sideband property should return a Parameters object containing only the sideband
        parameters."""
        if self.params is None:
            self.skipTest("Skipping: config failed to load")
        from_sideband_params = self.params.from_sideband
        self.assertIsInstance(from_sideband_params, Parameters)

    def test_from_sideband_parameters_are_identified(self):
        """The from_sideband property should correctly identify which parameters are from sideband
        based on the config."""
        if self.params is None:
            self.skipTest("Skipping: config failed to load")
        from_sideband_params = self.params.from_sideband
        self.assertEqual(len(from_sideband_params.names), 1)
        self.assertIn("nr_rate_multiplier", from_sideband_params.names)
        self.assertNotIn("er_rate_multiplier", from_sideband_params.names)

    def test_from_sideband_parameters_have_n_sideband_set(self):
        """The from_sideband parameters should have n_sideband and uncertainty set according to the
        config."""
        if self.params is None:
            self.skipTest("Skipping: config failed to load")
        from_sideband_params = self.params.from_sideband
        nr_param = from_sideband_params["nr_rate_multiplier"]
        self.assertTrue(nr_param.from_sideband)
        self.assertEqual(nr_param.n_sideband, 5)
        self.assertEqual(nr_param.uncertainty, 5)
