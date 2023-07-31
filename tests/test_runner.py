from os import remove
import pytest
from unittest import TestCase

from alea.utils import load_yaml
from alea.runner import Runner
from .test_gaussian_model import gaussian_model_parameter_definition


@pytest.mark.usefixtures('rm_cache')
class TestRunner(TestCase):
    """Test of the Runner class"""

    @classmethod
    def setUp(cls):
        """Initialise the Runner instance"""
        cls.runner_config = load_yaml('unbinned_wimp_running.yaml')
        cls.model_config = load_yaml(cls.runner_config['statistical_model_config'])
        cls.toydata_file = 'simple_data.h5'
        cls.output_file = 'test_toymc.h5'

    def set_gaussian_runner(self, toydata_mode='generate_and_write'):
        """Set a new runner instance with GaussianModel"""
        self.runner = Runner(
            statistical_model='alea.examples.gaussian_model.GaussianModel',
            poi='mu',
            hypotheses=['free', 'null', 'true'],
            n_mc=10,
            generate_values={'mu': 1., 'sigma': 1.},
            parameter_definition=gaussian_model_parameter_definition,
            toydata_mode=toydata_mode,
            toydata_file=self.toydata_file,
            output_file=self.output_file,
        )

    def set_blueice_runner(self, toydata_mode='generate_and_write'):
        """Set a new runner instance with BlueiceExtendedModel"""
        # TODO: interpret the config file after submitter class is implemented
        parameter_zvc = self.runner_config['computation']['discovery_power']
        self.runner = Runner(
            statistical_model=self.runner_config['statistical_model'],
            poi=self.runner_config['poi'],
            hypotheses=parameter_zvc['parameters_in_common']['hypotheses'],
            n_mc=10,
            generate_values={'wimp_rate_multiplier': 1.0},
            parameter_definition=self.model_config['parameter_definition'],
            likelihood_config=self.model_config['likelihood_config'],
            toydata_mode=toydata_mode,
            toydata_file=self.toydata_file,
            output_file=self.output_file,
        )

    def test_runners(self):
        """Test of the toy_simulation and write_output method"""
        set_runners = [self.set_gaussian_runner]
        # set_runners = [self.set_blueice_runner, self.set_gaussian_runner]
        for set_runner in set_runners:
            # test toydata_mode generate_and_write
            set_runner()
            self.runner.run()
            remove(self.output_file)

            # test toydata_mode read
            set_runner(toydata_mode='read')
            self.runner.run()
            remove(self.toydata_file)
            remove(self.output_file)
