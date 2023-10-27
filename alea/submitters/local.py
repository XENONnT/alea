import os
import json
import shlex
import warnings
import subprocess
import itertools
import operator
from functools import reduce
from copy import deepcopy
from typing import List, Dict, Any, Optional, Callable, cast

import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from inference_interface import toyfiles_to_numpy

from alea.runner import Runner
from alea.submitter import Submitter
from alea.utils import (
    load_json,
    asymptotic_critical_value,
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
        kwargs = Submitter.runner_kwargs_from_script(shlex.split(script)[2:])
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
            if llrs.min() < 0.0:
                self.logging.warning(
                    f"The lowest log likelihood ratio is negative {llrs.min():.2e}, "
                    f"total fraction of negative log likelihood ratio is "
                    f"{(llrs < 0.0).sum() / len(llrs):.2f}, "
                    f"the median if negative log likelihood ratios is {np.median(llrs[llrs < 0.0]):.2e}, "
                    f"there might be a problem in your fitting.",
                )
            if len(llrs) < 1000:
                self.logging.warning(
                    "The number of toys is less than 1000, the threshold might not be accurate!",
                )

            # update poi according to poi_expectation
            runner_args["statistical_model_args"].pop("limit_threshold", None)
            runner = Runner(**runner_args)
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
    def build_interpolator(
        poi,
        threshold,
        generate_values,
        nominal_values,
        confidence_level,
    ):
        """Build interpolator from the limit_threshold file.

        Args:
            poi (str): parameter of interest
            generate_values (dict): generate values assumed to be in the limit_threshold file,
                but is actually hypothesis. It should not contain poi or poi_expectation.
            nominal_values (dict): nominal values of parameters
            confidence_level (float): confidence level
            threshold (dict): threshold read directly from limit_threshold file

        Raises:
            ValueError: if the limit_threshold file does not contain the thresholds for the
                generate_values

        """
        names = list(generate_values.keys())
        inputs = threshold.values()
        # filter out the inputs with different nominal_values
        inputs = [i for i in inputs if i["hashed_keys"]["nominal_values"] == nominal_values]
        # filter out the inputs with different confidence_level
        inputs = [i for i in inputs if i["hashed_keys"]["confidence_level"] == confidence_level]
        # filter out the inputs with different generate_values keys
        inputs = [
            i for i in inputs if set(i["hashed_keys"]["generate_values"].keys()) == set(names)
        ]

        if len(inputs) == 0:
            raise ValueError(
                f"limit_threshold file does not contain "
                f"any threshold for nominal_values {nominal_values} "
                f"confidence_level {confidence_level}, and generate_values keys {names}!"
            )

        # get poi list
        poi_values = inputs[0][poi]
        if any([set(i[poi]) != set(poi_values) for i in inputs]):
            raise ValueError(
                f"poi {poi} lists with nominal_values {nominal_values} and "
                f"confidence_level {confidence_level} "
                "are not all the same in the limit_threshold file!"
            )

        # in the following code, we assume the generate_values are the same in the limit_threshold
        # and names, points, bounds, within have same length, each element corresponds to one name

        # deduce dimension of interpolator and collect all threshold into values
        points = []
        for n in names:
            points.append(
                np.unique([i["hashed_keys"]["generate_values"][n] for i in inputs]).tolist()
            )
        # check if the poi_values have the same length in the limit_threshold file
        if any([len(i[poi]) != len(poi_values) for i in inputs]):
            raise ValueError(
                f"poi {poi} lists with nominal_values {nominal_values} and "
                f"confidence_level {confidence_level} "
                "do not the same length in the limit_threshold file!"
            )
        # check if limit_threshold file contains enough threshold
        size = reduce(operator.mul, [len(p) for p in points])
        if len(inputs) != size:
            raise ValueError(
                f"limit_threshold file does not contain "
                f"enough threshold for nominal_values {nominal_values} "
                f"confidence_level {confidence_level}, and generate_values keys {names}!"
            )
        # build the values array for interpolator
        values = np.empty((*[len(p) for p in points], len(poi_values)), dtype=float)
        for p in itertools.product(*points):
            indices = [pi.index(pii) for pii, pi in zip(p, points)]
            values[indices] = [
                i["threshold"]
                for i in inputs
                if i["hashed_keys"]["generate_values"] == dict(zip(names, p))
            ]

        interpolator = RegularGridInterpolator(
            points + [poi_values], values, method="linear", bounds_error=True
        )
        return interpolator

    @staticmethod
    def get_confidence_interval_thresholds(
        poi,
        hypotheses_values,
        limit_threshold,
        nominal_values,
        confidence_interval_kind,
        confidence_level,
        limit_threshold_interpolation,
        asymptotic_dof: Optional[int] = 1,
    ):
        """Get confidence interval threshold function from limit_threshold file. If the
        limit_threshold file does not contain the threshold, it will interpolate the threshold from
        the existing threshold, using the RegularGridInterpolator, so in this case the threshold is
        not exact.

        Args:
            poi (str): parameter of interest
            hypotheses_values (list): hypotheses values for statistical model,
                only contains Dict[str, float]
            limit_threshold (str): path to the limit_threshold file
            nominal_values (dict): nominal values of parameters
            confidence_level (float): confidence level
            limit_threshold_interpolation (bool): whether to interpolate the threshold from the
                existing threshold, if the limit_threshold file does not contain the threshold
            asymptotic_dof (int, optional (default=1)):
                degrees of freedom for asymptotic critical value

        """

        if limit_threshold is None:
            return [None] * len(hypotheses_values)

        threshold = load_json(limit_threshold)

        func_list: List[Optional[Callable]] = []
        for i_hypo in range(len(hypotheses_values)):
            hypothesis = hypotheses_values[i_hypo]

            # keys for hashing, should be in limit_threshold
            _nominal_values = deepcopy(nominal_values) if nominal_values else {}
            _generate_values = deepcopy(hypothesis) if hypothesis else {}
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

            if (threshold_key not in threshold) and (not limit_threshold_interpolation):
                raise ValueError(
                    f"Looking for hashed_keys {hashed_keys}, but limit_threshold file "
                    f"{limit_threshold} does not contain "
                    f"{threshold_key}! Please check the limit_threshold file or set "
                    f"limit_threshold_interpolation as true!"
                )

            # get the poi_values and threshold_values for interp1d
            interpolator = None
            if threshold_key in threshold:
                # if threshold_key in threshold, get the poi_values and threshold_values directly
                poi_values = threshold[threshold_key][poi]
                threshold_values = threshold[threshold_key]["threshold"]
            else:
                # if threshold_key not in threshold,
                # interpolate the threshold from the existing threshold
                if len(hashed_keys["generate_values"]) == 0:
                    warnings.warn(
                        f"If hypothesis {hypothesis} does not container any values except "
                        f"poi, the nominal values should be in limit_threshold, so that "
                        f"the threshold can be interpolated from the existing threshold."
                    )
                    func_list.append(None)
                    continue

                if interpolator is None:
                    names = list(hashed_keys["generate_values"].keys())
                    interpolator = NeymanConstructor.build_interpolator(
                        poi,
                        threshold,
                        hashed_keys["generate_values"],
                        hashed_keys["nominal_values"],
                        confidence_level,
                    )
                    poi_values = interpolator.grid[-1]

                if set(hashed_keys["generate_values"].keys()) != set(names):
                    raise ValueError(
                        f"When using limit_threshold_interpolation, "
                        f"the keys of hypotheses should be "
                        f"the same for all hypotheses(expect str hypotheses)! "
                        f"But previously find {set(names)} and now find "
                        f"{set(hashed_keys['generate_values'].keys())}!"
                    )
                pts = [list(hashed_keys["generate_values"].values()) + [p] for p in poi_values]
                try:
                    threshold_values = interpolator(pts)
                except ValueError:
                    raise ValueError(
                        f"Interpolation failed for hypothesis. "
                        f"Maybe the limit_threshold file does not contain enough threshold, "
                        f"so that {hashed_keys['generate_values']} is out of bounds. "
                        f"Please check the limit_threshold file!"
                    )

            # if out of bounds, return the asymptotic critical value
            func = interp1d(
                poi_values,
                threshold_values,
                bounds_error=False,
                fill_value=asymptotic_critical_value(
                    confidence_interval_kind, confidence_level, asymptotic_dof
                ),
            )
            func_list.append(func)

        # check if the length of hypotheses and func_list are the same
        if len(hypotheses_values) != len(func_list):
            raise ValueError(
                "Something wrong with the length of hypotheses and func_list, "
                "please check the code!"
            )
        return func_list
