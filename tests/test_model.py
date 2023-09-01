from unittest import TestCase

from alea import StatisticalModel


class TestStatisticalModel(TestCase):
    """Test of the GaussianModel class."""

    def test_statistical_model(self):
        with self.assertRaises(
            RuntimeError, msg="Should raise error when directly instantiating StatisticalModel"
        ):
            StatisticalModel()
