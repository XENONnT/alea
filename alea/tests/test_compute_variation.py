import unittest
from alea.utils import compute_variations, compute_parameters_to_vary, compute_parameters_to_zip, read_config
from alea.toymc_running import toymc_to_sbatch_call_array_update
from alea.toymc_running import toymc_to_sbatch_call_array, compute_neyman_thresholds_update
import os
import pkg_resources


class TestComputeVariation(unittest.TestCase):
    def test_parameters_to_vary(self):
        test_case = {"var1": [1, 2, 3], "var2": [4, 5, 6]}

        list_of_varied_dict = compute_parameters_to_vary(
            parameters_to_vary=test_case)

        result = [(1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 4),
                  (3, 5), (3, 6)]
        for mydict, res in zip(list_of_varied_dict, result):
            self.assertEqual(tuple(mydict.values()), res)

        test_case = {"generate_args": {"var1": [1, 2, 4], "var2": [4, 5, 6]}}

        list_of_varied_dict = compute_parameters_to_vary(
            parameters_to_vary=test_case)

        result = [{
            'generate_args': {
                'var1': 1,
                'var2': 4
            }
        }, {
            'generate_args': {
                'var1': 1,
                'var2': 5
            }
        }, {
            'generate_args': {
                'var1': 1,
                'var2': 6
            }
        }, {
            'generate_args': {
                'var1': 2,
                'var2': 4
            }
        }, {
            'generate_args': {
                'var1': 2,
                'var2': 5
            }
        }, {
            'generate_args': {
                'var1': 2,
                'var2': 6
            }
        }, {
            'generate_args': {
                'var1': 4,
                'var2': 4
            }
        }, {
            'generate_args': {
                'var1': 4,
                'var2': 5
            }
        }, {
            'generate_args': {
                'var1': 4,
                'var2': 6
            }
        }]

        for item, res in zip(list_of_varied_dict, result):
            self.assertEqual(item, res)

        test_case = {"livetime": [[1, 2, 4], [1, 2, 3]], "var1": [1]}

        list_of_varied_dict = compute_parameters_to_vary(
            parameters_to_vary=test_case)

        result = [{
            'livetime': [1, 2, 4],
            "var1": 1
        }, {
            "livetime": [1, 2, 3],
            "var1": 1
        }]
        for item, res in zip(list_of_varied_dict, result):
            self.assertEqual(item, res)

    def test_parameters_to_zip(self):
        parameters_to_zip = {
            "var1": [1, 2, 3, 4],
            "var2": [4, 5, 6, 7],
            "generate_args": {
                "var1": [4, 5, 6, 7]
            }
        }

        zipped_dicts = compute_parameters_to_zip(
            parameters_to_zip=parameters_to_zip)
        result = [{
            'var1': 1,
            'var2': 4,
            'generate_args': {
                'var1': 4
            }
        }, {
            'var1': 2,
            'var2': 5,
            'generate_args': {
                'var1': 5
            }
        }, {
            'var1': 3,
            'var2': 6,
            'generate_args': {
                'var1': 6
            }
        }, {
            'var1': 4,
            'var2': 7,
            'generate_args': {
                'var1': 7
            }
        }]
        self.assertEqual(zipped_dicts, result)

    def test_compute_variations(self):
        test_zip = {
            "var1": [1, 2, 3, 4],
            "var2": [4, 5, 6, 7],
            "generate_args": {
                "var1": [4, 5, 6, 7]
            },
            "livetime": [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        }
        test_vary = {
            "generate_args": {
                "var2": [2, 3, 4]
            },
            "livetime": [[1, 2, 4], [1, 2, 3]],
            "var3": [1]
        }
        config_file = pkg_resources.resource_filename(
            "alea",
            "tests/test_configs/ll_nt_lowfield_v7_combination_test.yaml")
        config_data = read_config(config_file)

        computation = "discovery_power"
        merged_combinations = compute_variations(
            parameters_to_vary=config_data["computation"][computation].get(
                "parameters_to_vary", {}),
            parameters_to_zip=config_data["computation"][computation].get(
                "parameters_to_zip", {}),
            parameters_in_common=config_data["computation"][computation].get(
                "parameters_in_common", {}))

    def test_toymc_to_sbatch_call_array_update(self):
        config_file = pkg_resources.resource_filename(
            "alea",
            "tests/test_configs/ll_nt_lowfield_v7_combination_test.yaml")
        config_data = read_config(config_file)

        computation = "discovery_power"
        fnames, calls = toymc_to_sbatch_call_array_update(
            parameters_to_vary=config_data["computation"][computation].get(
                "parameters_to_vary", {}),
            parameters_to_zip=config_data["computation"][computation].get(
                "parameters_to_zip", {}),
            parameters_in_common=config_data["computation"][computation].get(
                "parameters_in_common", {}))

        fnames_old, calls_old = toymc_to_sbatch_call_array(
            parameters_to_vary=config_data["computation"][computation].get(
                "parameters_to_vary", {}),
            parameters_to_zip=config_data["computation"][computation].get(
                "parameters_to_zip", {}),
            parameters_in_common=config_data["computation"][computation].get(
                "parameters_in_common", {}))

        fnames_old.sort(), fnames.sort()
        #  self.assertTrue(fnames_old, fnames)
        #  calls.sort(), calls_old.sort()
        #  self.assertTrue(calls, calls_old)
        self.assertEqual(len(calls), len(calls_old))
        self.assertEqual(len(fnames), len(fnames_old))

    def test_compute_neyman(self):
        config_file = pkg_resources.resource_filename(
            "alea",
            "tests/test_configs/ll_nt_lowfield_v7_combination_test.yaml")
        config_data = read_config(config_file)
        parameters_in_common = config_data['computation']["threshold"][
            "parameters_in_common"]
        parameters_to_zip = config_data['computation']["threshold"][
            "parameters_to_zip"]
        parameters_to_vary = config_data['computation']["threshold"][
            "parameters_to_vary"]
        output_filename = parameters_in_common.get("output_filename")
        file_name_pattern = output_filename.split(
            ".hdf5")[0] + "_{n_batch:d}.hdf5"
        file_name_pattern = os.path.join(config_data["outputfolder"],
                                         file_name_pattern)
        mydict = compute_neyman_thresholds_update(
            file_name_pattern=file_name_pattern,
            parameters_in_common=parameters_in_common,
            parameters_to_zip=parameters_to_zip,
            parameters_to_vary=parameters_to_vary,
            return_to_dict=False)

        print(mydict)


if __name__ == "__main__":
    unittest.main()
