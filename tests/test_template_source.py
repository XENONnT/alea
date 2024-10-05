from unittest import TestCase

import shutil
from inference_interface import template_to_multihist, multihist_to_template
from alea.models import BlueiceExtendedModel
from alea.utils import load_yaml


class TestTemplateSource(TestCase):
    """Test of the TemplateSource class."""

    def test_init_templates(self):
        """Test whether we can initialize template sources."""
        model_configs = load_yaml("unbinned_wimp_statistical_model_template_source_test.yaml")
        parameter_definition = model_configs["parameter_definition"]
        likelihood_config = model_configs["likelihood_config"]
        model = BlueiceExtendedModel(parameter_definition, likelihood_config)
        model.nominal_expectation_values

    def test_wrong_analysis_space(self):
        """Test whether initializing with a wrong analysis_space raises error."""
        model_configs = load_yaml("unbinned_wimp_statistical_model_template_source_test.yaml")
        parameter_definition = model_configs["parameter_definition"]
        likelihood_config = model_configs["likelihood_config"]
        # Change the analysis space to a wrong one
        space = likelihood_config["likelihood_terms"][0]["analysis_space"]
        # additive mismatch in cs1
        space[0]["cs1"] = "np.linspace(1, 101, 51)"
        space[1]["cs2"] = "np.geomspace(100, 100000, 51)"
        with self.assertRaises(AssertionError):
            _ = BlueiceExtendedModel(parameter_definition, likelihood_config)

        # multiplicative mismatch in cs2
        space[0]["cs1"] = "np.linspace(0, 100, 51)"
        space[1]["cs2"] = "np.geomspace(101, 101000, 51)"
        with self.assertRaises(AssertionError):
            _ = BlueiceExtendedModel(parameter_definition, likelihood_config)

        # If the dimensions are wrong we should get ValueError
        space[0]["cs1"] = "np.linspace(0, 100, 50)"
        space[1]["cs2"] = "np.geomspace(100, 100000, 51)"
        with self.assertRaises(ValueError):
            _ = BlueiceExtendedModel(parameter_definition, likelihood_config)

    def test_hash_updates_on_template_change(self):
        """Test whether the hash of the template changes when the file is modified."""
        config_path = "unbinned_wimp_statistical_model_template_source_test.yaml"
        model = BlueiceExtendedModel.from_config(config_path)

        hash_er_sr2 = model.likelihood_list[0].base_model.sources[0].hash
        hash_wimp_sr2 = model.likelihood_list[0].base_model.sources[1].hash
        hash_er_sr3 = model.likelihood_list[1].base_model.sources[0].hash
        hash_wimp_sr3 = model.likelihood_list[1].base_model.sources[1].hash

        # modify one of the ER templates
        path = model.likelihood_list[1].base_model.sources[0].config["templatename"]
        path = path.format(er_band_shift=0)
        backup_path = path + ".backup"
        shutil.copy(path, backup_path)
        h = template_to_multihist(path, "er_template")
        h.histogram *= 2
        multihist_to_template([h], path, ["er_template"])

        # check that the hash of the ER sources has changed
        model = BlueiceExtendedModel.from_config(config_path)
        self.assertNotEqual(hash_er_sr2, model.likelihood_list[0].base_model.sources[0].hash)
        self.assertNotEqual(hash_er_sr3, model.likelihood_list[1].base_model.sources[0].hash)
        self.assertEqual(hash_wimp_sr2, model.likelihood_list[0].base_model.sources[1].hash)
        self.assertEqual(hash_wimp_sr3, model.likelihood_list[1].base_model.sources[1].hash)
        # restore the original template
        shutil.move(backup_path, path)

        # modify the WIMP template
        path = model.likelihood_list[1].base_model.sources[1].config["templatename"]
        backup_path = path + ".backup"
        shutil.copy(path, backup_path)
        h = template_to_multihist(path, "wimp_template")
        h.histogram *= 2
        multihist_to_template([h], path, ["wimp_template"])

        # check that the hash of the WIMP sources has changed
        model = BlueiceExtendedModel.from_config(config_path)
        self.assertEqual(hash_er_sr2, model.likelihood_list[0].base_model.sources[0].hash)
        self.assertEqual(hash_er_sr3, model.likelihood_list[1].base_model.sources[0].hash)
        self.assertNotEqual(hash_wimp_sr2, model.likelihood_list[0].base_model.sources[1].hash)
        self.assertNotEqual(hash_wimp_sr3, model.likelihood_list[1].base_model.sources[1].hash)
        # restore the original template
        shutil.move(backup_path, path)
