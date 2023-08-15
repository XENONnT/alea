import os
import json
import shlex
import subprocess
from typing import List, Dict, Any, cast

import numpy as np
from inference_interface import toyfiles_to_numpy

from alea.runner import Runner
from alea.submitter import Submitter
from alea.utils import deterministic_hash


class SubmitterLocal(Submitter):
    """Submitter for local machine."""

    def __init__(self, *args, **kwargs):
        """Initialize the SubmitterLocal class."""
        self.local_configurations = kwargs.get("local_configurations", {})
        self.template_path = self.local_configurations.pop("template_path", None)
        super().__init__(*args, **kwargs)

    @staticmethod
    def initialized_runner(script: str):
        kwargs = Submitter.init_runner_from_args_string(shlex.split(script)[1:])
        runner = Runner(**kwargs)
        return runner

    def submit(self):
        """Run job in subprocess locally.

        If debug is True, only return the first instance of Runner.

        """
        for _, (script, _) in enumerate(self.computation_tickets_generator()):
            if self.debug:
                print(script)
                return self.initialized_runner(script)
            subprocess.call(shlex.split(script))


class NeymanConstructor(SubmitterLocal):
    """Neyman threshold constructor.

    Will not really submit a job.

    """

    def submit(
        self,
        free_name: str = "free",
        true_name: str = "true",
        confidence_levels: List[float] = [0.8, 0.9, 0.95],
    ):
        """Read the likelihood ratio from the output files and calculate the Neyman threshold.

        Args:
            free_name: the name of the free hypothesis
            true_name: the name of the true hypothesis
            confidence_levels: the confidence levels to calculate the threshold

        """
        script = next(self.computation_tickets_generator())[0]
        runner = self.initialized_runner(script)

        threshold = cast(Dict[str, Any], {})
        for runner_args in self.merged_arguments_generator():
            # prepare the needed nominal_values and generate_values
            nominal_values = runner_args["nominal_values"]
            generate_values = runner_args["generate_values"]
            needed_kwargs = {
                **nominal_values,
                **generate_values,
            }

            # read the likelihood ratio
            output_file = runner_args["output_file"]
            # add a * to the output_file to read all the files
            fpat_split = os.path.splitext(output_file)
            output_file = fpat_split[0] + "*" + fpat_split[1]
            results = toyfiles_to_numpy(output_file.format(**needed_kwargs))
            llfree = results[free_name]["ll"]
            lltrue = results[true_name]["ll"]
            llrs = 2.0 * (llfree - lltrue)

            # update poi according to poi_expectation
            if runner.input_poi_expectation:
                poi_expectation = generate_values.get("poi_expectation")
                generate_values = runner.update_poi(self.poi, generate_values, nominal_values)
            else:
                expectation_values = runner.model.get_expectation_values(
                    **{**nominal_values, **generate_values}
                )
                # in some rare cases the poi is not a rate multiplier
                # then the poi_expectation is not in the nominal_expectation_values
                component = self.poi.replace("_rate_multiplier", "")
                poi_expectation = expectation_values.get(component, None)
            poi_value = generate_values.pop(self.poi)

            # make sure no poi and poi_expectation in the hashed_keys
            generate_values.pop(self.poi, None)
            generate_values.pop("poi_expectation", None)
            nominal_values.pop(self.poi, None)
            nominal_values.pop("poi_expectation", None)

            for confidence_level in confidence_levels:
                q_llr = np.percentile(llrs, 100.0 * confidence_level).item()
                hashed_keys = {
                    "poi": self.poi,
                    "nominal_values": nominal_values,
                    "generate_values": generate_values,
                    "confidence_level": confidence_level,
                }
                threshold_key = deterministic_hash(hashed_keys)
                if threshold_key not in threshold:
                    threshold_value = {
                        "hashed_keys": hashed_keys,
                        self.poi: [],
                        "threshold": [],
                        "poi_expectation": [],
                    }
                    threshold[threshold_key] = threshold_value
                threshold[threshold_key][self.poi].append(poi_value)
                threshold[threshold_key]["threshold"].append(q_llr)
                threshold[threshold_key]["poi_expectation"].append(poi_expectation)

        # sort the threshold based on the elements of poi
        for k, v in threshold.items():
            paired = zip(v[self.poi], v["threshold"], v["poi_expectation"])
            # sort the pairs based on the elements of poi
            sorted_pairs = sorted(paired, key=lambda x: x[0])
            threshold[k][self.poi] = [x[0] for x in sorted_pairs]
            threshold[k]["threshold"] = [x[1] for x in sorted_pairs]
            threshold[k]["poi_expectation"] = [x[2] for x in sorted_pairs]

        # save the threshold into a json file
        if os.path.splitext(runner_args["limit_threshold"])[-1] != ".json":
            raise ValueError("The limit_threshold file should be a json file.")
        with open(runner_args["limit_threshold"], mode="w") as f:
            json.dump(threshold, f, indent=4)
