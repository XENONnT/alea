from unittest import TestCase

from alea.utils import (
    get_analysis_space,
    formatted_to_asterisked,
    can_expand_grid,
    expand_grid_dict,
    deterministic_hash,
)


class TestUtils(TestCase):
    """Test of the alea.utils."""

    def test_get_analysis_space(self):
        """Test of the get_analysis_space function."""
        analysis_space = [
            {"a": "np.arange(3)"},
            {"b": [0, 1, 2]},
            {"c": "0, 1, 2"},
            {"d": "0 1 2"},
        ]
        get_analysis_space(analysis_space)

    def test_formatted_to_asterisked(self):
        """Test of the formatted_to_asterisked function."""
        self.assertEqual(formatted_to_asterisked("a_{a:.2f}_b_{b:d}"), "a_*_b_*")
        self.assertEqual(formatted_to_asterisked("a_{a:.2f}_b_{b:d}", wildcards="a"), "a_*_b_{b:d}")

    def test_can_expand_grid(self):
        """Test of the can_expand_grid function."""
        self.assertTrue(can_expand_grid({"a": [1, 2], "b": [3, 4]}))

    def test_expand_grid_dict(self):
        """Test of the expand_grid_dict function."""
        self.assertEqual(
            expand_grid_dict(["free", {"a": 1, "b": 3}, {"a": [1, 2], "b": [3, 4]}]),
            [
                "free",
                {"a": 1, "b": 3},
                {"a": 1, "b": 3},
                {"a": 1, "b": 4},
                {"a": 2, "b": 3},
                {"a": 2, "b": 4},
            ],
        )

    def test_deterministic_hash(self):
        """Test of the deterministic_hash function."""
        self.assertEqual(deterministic_hash({"a": 1, "b": 2}), "shhkapn4q7")
