import os
import json
import shlex
import subprocess
import itertools
from copy import deepcopy
from typing import List, Dict, Any, cast

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from inference_interface import toyfiles_to_numpy

from alea.runner import Runner
from alea.submitter import Submitter
from alea.utils import (
    load_json,
    within_limits,
    confidence_interval_critical_value,
    deterministic_hash,
)


class SubmitterLocal(Submitter):
    """Submitter for local machine."""

    def __init__(self, *args, **kwargs):
        """Initialize the SubmitterLocal class."""
        self.local_configurations = kwargs.get("local_configurations", {})
        self.template_path = self.local_configurations.pop("template_path", None)
        super().__init__(*args, **kwargs)

    @staticmethod
    def initialized_runner(script: str, pop_limit_threshold: bool = False):
        """Initialize a Runner from a script.

        Args:
            script: the script to initialize the Runner
            pop_limit_threshold: whether to pop the limit_threshold from the
                statistical_model_args, this is needed for the NeymanConstructor,
                to initialize runner when the limit_threshold does not exist yet.

        """
        kwargs = Submitter.init_runner_from_args_string(shlex.split(script)[2:])
        if pop_limit_threshold:
            kwargs["statistical_model_args"].pop("limit_threshold", None)
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

    allowed_special_args = ["free_name", "true_name", "confidence_levels"]

    def submit(
        self,
        free_name: str = "free",
        true_name: str = "true",
        confidence_levels: List[float] = [0.8, 0.9, 0.95],
    ):
        """Read the likelihood ratio from the output files and calculate the Neyman threshold. The
        threshold will be saved into a json file. The threshold will be sorted based on the elements
        of poi.

        Args:
            free_name: the name of the free hypothesis
            true_name: the name of the true hypothesis
            confidence_levels: the confidence levels to calculate the threshold

        Example:
            >>> data = json.load(open("limit_threshold.json")); print(json.dumps(data, indent=4))
            {
                "oyldh6bczx": {
                    "hashed_keys": {
                        "poi": "wimp_rate_multiplier",
                        "nominal_values": {
                            "wimp_mass": 0,
                            "livetime": 0.0
                        },
                        "generate_values": {},
                        "confidence_level": 0.9
                    },
                    "wimp_rate_multiplier": [
                        0.0,
                        1.0,
                        2.0
                    ],
                    "threshold": [
                        0.0,
                        0.0,
                        0.0
                    ],
                    "poi_expectation": [
                        null,
                        null,
                        null
                    ]
                },
            }

        """

        # overwrite the free_name, true_name and confidence_levels from the runner_args
        runner_args = next(self.merged_arguments_generator())
        if "free_name" in runner_args:
            free_name = runner_args["free_name"]
            self.logging.info(f"Overwrite free_name to {free_name}.")
        if "true_name" in runner_args:
            true_name = runner_args["true_name"]
            self.logging.info(f"Overwrite true_name to {true_name}.")
        if "confidence_levels" in runner_args:
            confidence_levels = runner_args["confidence_levels"]
            self.logging.info(f"Overwrite confidence_levels to {confidence_levels}.")

        # extract limit_threshold from the statistical_model_args
        limit_threshold = runner_args["statistical_model_args"].get("limit_threshold", None)
        if limit_threshold is None:
            raise ValueError("Please specify the limit_threshold at in_common.")
        if os.path.splitext(limit_threshold)[-1] != ".json":
            raise ValueError("The limit_threshold file should be a json file.")

        # initialize the runner
        script = next(self.computation_tickets_generator())[0]
        runner = self.initialized_runner(script, pop_limit_threshold=True)

        # calculate the threshold, iterate over the output files
        threshold = cast(Dict[str, Any], {})
        for runner_args in self.merged_arguments_generator():
            # check if the free_name, true_name and confidence_levels are consistent
            message = (
                " is not consistent in one of your runner arguments, "
                "please only specify it in in_common."
            )
            if runner_args.pop("free_name", free_name) != free_name:
                raise ValueError("free_name" + message)
            if runner_args.pop("true_name", true_name) != true_name:
                raise ValueError("true_name" + message)
            new_confidence_levels = runner_args.pop("confidence_levels", confidence_levels)
            mask = any([n_c != c for n_c, c in zip(new_confidence_levels, confidence_levels)])
            mask |= len(new_confidence_levels) != len(confidence_levels)
            if mask:
                raise ValueError("confidence_levels" + message)

            # prepare the needed nominal_values and generate_values
            nominal_values = runner_args["nominal_values"]
            generate_values = runner_args["generate_values"]
            needed_kwargs = {
                **nominal_values,
                **generate_values,
            }

            # read the likelihood ratio
            output_filename = runner_args["output_filename"]
            # add a * to the output_filename to read all the files
            fpat_split = os.path.splitext(output_filename)
            output_filename = fpat_split[0] + "*" + fpat_split[1]
            results = toyfiles_to_numpy(output_filename.format(**needed_kwargs))
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

            # calculate the threshold given different confidence levels
            for confidence_level in confidence_levels:
                q_llr = np.percentile(llrs, 100.0 * confidence_level).item()
                if q_llr < 0.0:
                    self.logging.warning(
                        f"The threshold is negative ({q_llr}) for confidence_level "
                        f"{confidence_level}, there might be a problem in your fitting."
                    )
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
        with open(limit_threshold, mode="w") as f:
            json.dump(threshold, f, indent=4)
        print(f"Saving {limit_threshold}")

    @staticmethod
    def get_confidence_interval_threshold(
        poi,
        statistical_model_args,
        generate_values,
        nominal_values,
        confidence_interval_kind,
        confidence_level,
    ):
        """Get confidence interval threshold function from limit_threshold file. If the
        limit_threshold file does not contain the threshold, it will interpolate the threshold from
        the existing threshold, using the RegularGridInterpolator, so in this case the threshold is
        not exact.

        Args:
            poi (str): parameter of interest
            statistical_model_args (dict): arguments for statistical model
            generate_values (dict): generate values of toydata,
                it can contain "poi_expectation"
            nominal_values (dict): nominal values of parameters
            confidence_level (float): confidence level

        """
        if "limit_threshold" not in statistical_model_args:
            return None

        # keys for hashing, should be in limit_threshold
        _nominal_values = deepcopy(nominal_values) if nominal_values else {}
        _generate_values = deepcopy(generate_values) if generate_values else {}
        hashed_keys = {
            "poi": poi,
            "nominal_values": deepcopy(_nominal_values),
            "generate_values": deepcopy(_generate_values),
            "confidence_level": confidence_level,
        }

        # make sure no poi and poi_expectation in the hashed_keys
        hashed_keys["generate_values"].pop(poi, None)
        hashed_keys["generate_values"].pop("poi_expectation", None)
        hashed_keys["nominal_values"].pop(poi, None)
        hashed_keys["nominal_values"].pop("poi_expectation", None)
        threshold_key = deterministic_hash(hashed_keys)
        limit_threshold = statistical_model_args["limit_threshold"]
        threshold = load_json(limit_threshold)

        limit_threshold_interpolation = statistical_model_args.get(
            "limit_threshold_interpolation", False
        )
        if (threshold_key not in threshold) and (not limit_threshold_interpolation):
            raise ValueError(
                f"limit_threshold file {statistical_model_args['limit_threshold']} "
                f"does not contain {threshold_key}, Looking for hashed_keys {hashed_keys}! "
                f"Please check the limit_threshold file or set limit_threshold_interpolation "
                f"as true!"
            )

        if threshold_key in threshold:
            poi_values = threshold[threshold_key][poi]
            threshold_values = threshold[threshold_key]["threshold"]
        else:
            inputs = threshold.values()
            # filter out the inputs with different nominal_values
            nv = hashed_keys["nominal_values"]
            inputs = [i for i in inputs if i["hashed_keys"]["nominal_values"] == nv]
            # filter out the inputs with different confidence_level
            inputs = [i for i in inputs if i["hashed_keys"]["confidence_level"] == confidence_level]

            if len(inputs) == 0:
                raise ValueError(
                    f"limit_threshold file {limit_threshold} does not contain "
                    f"the threshold for nominal_values {nv} and confidence_level "
                    f"{confidence_level}!"
                )

            # get poi list
            poi_values = inputs[0][poi]
            if any([set(i[poi]) != set(poi_values) for i in inputs]):
                raise ValueError(
                    f"poi list is not the same in the limit_threshold file {limit_threshold}!"
                )

            # get generate_values list
            names = list(inputs[0]["hashed_keys"]["generate_values"].keys())
            if any([set(i["hashed_keys"]["generate_values"].keys()) != set(names) for i in inputs]):
                raise ValueError(
                    f"generate_values list is not the same in the "
                    f"limit_threshold file {limit_threshold}!"
                )

            # collect all threshold into values
            points = []
            for g in names:
                points.append(
                    np.unique([i["hashed_keys"]["generate_values"][g] for i in inputs]).tolist()
                )
            values = np.empty((*[len(g) for g in points], len(poi_values)), dtype=float)
            for g in itertools.product(*points):
                indices = [pi.index(gi) for gi, pi in zip(g, points)]
                values[indices] = [
                    i["threshold"]
                    for i in inputs
                    if i["hashed_keys"]["generate_values"] == dict(zip(names, g))
                ]

            # interpolate the threshold
            bounds = [[min(p), max(p)] for p in points]
            within = [within_limits(_generate_values[n], b) for n, b in zip(names, bounds)]
            if not all(within):
                out_names = [n for n, w in zip(names, within) if not w]
                out_bounds = [b for b, w in zip(bounds, within) if not w]
                raise ValueError(
                    f"generate_values {out_names} are out of bounds "
                    f"{dict(zip(out_names, out_bounds))}."
                )
            interpolator = RegularGridInterpolator(
                points + [poi_values], values, method="linear", bounds_error=True
            )
            pts = [[_generate_values[n] for n in names] + [p] for p in poi_values]
            threshold_values = interpolator(pts)

        # if out of bounds, return the asymptotic critical value
        func = interp1d(
            poi_values,
            threshold_values,
            bounds_error=False,
            fill_value=confidence_interval_critical_value(
                confidence_interval_kind, confidence_level
            ),
        )
        return func
