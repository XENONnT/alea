from os import remove
import pytest
from unittest import TestCase

from alea.utils import load_yaml
from alea.runner import Runner


@pytest.mark.usefixtures('rm_cache')
class TestRunner(TestCase):
    """Test of the Runner class"""

    @classmethod
    def setUp(cls):
        """Initialise the Runner instance"""
        cls.runner_config = load_yaml('unbinned_wimp_running.yaml')
        cls.model_config = load_yaml(cls.runner_config['statistical_model_config'])
        cls.set_new_runner(cls)

    def set_new_runner(self):
        """Set a new BlueiceExtendedModel instance"""
        # TODO: interpret the config file after submitter class is implemented
        parameter_zvc = self.runner_config['computation']['discovery_power']
        self.runner = Runner(
            statistical_model = self.runner_config['statistical_model'],
            poi = self.runner_config['poi'],
            hypotheses = parameter_zvc['parameters_in_common']['hypotheses'],
            n_mc = 10,
            generate_values = {'wimp_rate_multiplier': 1.0},
            parameter_definition = self.model_config['parameter_definition'],
            likelihood_config = self.model_config['likelihood_config'],
            toydata_mode = 'generate_and_write',
            toydata_file = 'simple_data.h5',
            output_file = 'test_toymc.h5',
        )

    def test_run(self):
        """Test of the toy_simulation and write_output method"""
        self.runner.run()
        remove('simple_data.h5')
        remove('test_toymc.h5')
