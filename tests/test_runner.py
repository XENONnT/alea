from os import remove
import inspect
import pytest
from unittest import TestCase

import numpy as np
from inference_interface import toyfiles_to_numpy

from alea.utils import load_yaml
from alea.runner import Runner

from .test_gaussian_model import gaussian_model_parameter_definition


COMPUTE_CONFIDENCE_INTERVAL = True


@pytest.mark.usefixtures("rm_cache")
class TestRunner(TestCase):
    """Test of the Runner class."""

    @classmethod
    def setUp(cls):
        """Initialise the Runner instance."""
        cls.running_config = load_yaml("unbinned_wimp_running.yaml")
        cls.toydata_filename = "simple_data.ii.h5"
        cls.output_filename = "test_toymc.ii.h5"
        cls.n_mc = 3

    def set_gaussian_runner(self, toydata_mode="generate_and_store"):
        """Set a new runner instance with GaussianModel."""
        self.runner = Runner(
            statistical_model="alea.examples.gaussian_model.GaussianModel",
            poi="mu",
            hypotheses=["free", "zero", "true"],
            n_mc=self.n_mc,
            generate_values={"mu": 1.0},
            nominal_values={"sigma": 1.0},
            parameter_definition=gaussian_model_parameter_definition,
            compute_confidence_interval=COMPUTE_CONFIDENCE_INTERVAL,
            toydata_mode=toydata_mode,
            toydata_filename=self.toydata_filename,
            output_filename=self.output_filename,
        )

    def set_blueice_runner(self, toydata_mode="generate_and_store"):
        """Set a new runner instance with BlueiceExtendedModel."""
        parameter_zvc = self.running_config["computation_options"]["discovery_power"]
        self.runner = Runner(
            statistical_model=self.running_config["statistical_model"],
            poi=self.running_config["poi"],
            hypotheses=parameter_zvc["in_common"]["hypotheses"],
            n_mc=self.n_mc,
            generate_values={"wimp_rate_multiplier": 1.0},
            statistical_model_config=self.running_config["statistical_model_config"],
            compute_confidence_interval=COMPUTE_CONFIDENCE_INTERVAL,
            toydata_mode=toydata_mode,
            toydata_filename=self.toydata_filename,
            output_filename=self.output_filename,
        )

    def test_runners(self):
        """Test of the simulate_and_fit and write_output method."""
        set_runners = [self.set_gaussian_runner, self.set_blueice_runner]
        for set_runner in set_runners:
            # test toydata_mode generate_and_store
            set_runner()
            self.runner.run()
            remove(self.output_filename)

            # test toydata_mode read
            set_runner(toydata_mode="read")
            self.runner.run()
            remove(self.toydata_filename)

            # check confidence interval computation
            if COMPUTE_CONFIDENCE_INTERVAL:
                results = toyfiles_to_numpy(self.runner._output_filename)
                mask = np.any(np.isnan(results["free"]["dl"]))
                mask &= np.any(np.isnan(results["free"]["ul"]))
                if mask:
                    raise ValueError("Confidence interval computation failed!")
            remove(self.output_filename)

    def test_init_signatures(self):
        """Test the signatures of the Runner.__init__"""
        (
            args,
            varargs,
            varkw,
            defaults,
            kwonlyargs,
            kwonlydefaults,
            annotations,
        ) = inspect.getfullargspec(Runner.__init__)
        if (len(annotations) != len(args[1:])) or (len(defaults) != len(args[1:])):
            raise ValueError(
                "The number of annotations and defaults of Runner.__init__ must be the same!"
            )
