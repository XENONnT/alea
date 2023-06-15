import unittest
import inspect
import itertools
import yaml
from binference.utils import VariationConvenience

from binference.likelihoods import ll_nt_from_config


class TestVariationConvenience(unittest.TestCase):

    def test_product(self):
        parameters_to_vary = {"livetime": [1.0, 2.0], "wimp_mass": [50]}
        signature = inspect.signature(ll_nt_from_config.InferenceObject)
        with self.assertRaises(SystemExit):
            varcon = VariationConvenience(parameters_to_vary=parameters_to_vary,
                                          parameters_to_zip={},
                                          parameters_in_common={},
                                          parameters_as_wildcards=[],
                                          generate_args_parameters=[],
                                          signature=signature
                                          )

        #  product = varcon.genereate_binference_input()
        #  for item in product:
        #      print(item)

    def test_parameter_splitting(self):
        parameters_to_vary = {"livetime": [1.0, 2.0], "wimp_mass": [50], "er_rate_multiplier": [95, 190], "s2_threshold": [300, 400]}
        generate_args_parameter = ["er_rate_multiplier", "AC_rate_multiplier"]
        signature = inspect.signature(ll_nt_from_config.InferenceObject)

        varcon = VariationConvenience(parameters_to_vary=parameters_to_vary,
                                      parameters_to_zip={},
                                      parameters_as_wildcards=[],
                                      parameters_in_common={},
                                      generate_args_parameters=generate_args_parameter,
                                      signature=signature
                                      )
        varcon.split_parameters()

        self.assertTrue(sorted(list(varcon.parameters_to_vary_in_likelihood.keys())) == sorted(["s2_threshold"]))
        self.assertTrue(sorted(list(varcon.parameters_to_vary_in_generate_args.keys())) == sorted(["er_rate_multiplier"]))
        self.assertTrue(sorted(list(varcon.parameters_to_vary_in_signature.keys())) == sorted(["wimp_mass", "livetime"]))
        self.assertTrue(len(varcon.parameters_to_vary) == len(varcon.parameters_to_vary_in_likelihood.keys()) + len(varcon.parameters_to_vary_in_generate_args.keys()) + len(varcon.parameters_to_vary_in_signature.keys()))

        varcon._create_zip_input()
        varcon._create_generate_args_input()
        varcon._create_likelihood_args_input()

        varcon.combined_zip_input()

        for key, value in varcon.zip_input.items():
            self.assertTrue(len(value) == 8)


if __name__ == "__main__":
    unittest.main()
