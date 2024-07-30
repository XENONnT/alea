import os
import json
import shlex
import warnings
import subprocess
import itertools
import operator
from glob import glob
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
    search_filename_pattern,
    get_metadata,
)


class SubmitterLocal(Submitter):
    """Submitter for local machine."""

    def __init__(self, *args, **kwargs):
        """Initialize the SubmitterLocal class."""
        self.local_configurations = kwargs.get("local_configurations", {})
        self.template_path = self.local_configurations.pop("template_path", None)
        self.combine_n_jobs = self.local_configurations.pop("combine_n_jobs", 1)
        self.first_i_batch = kwargs.pop("i_batch", 0)
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

    def submit(self, *args, **kwargs):
        """Run job in subprocess locally.

        If debug is True, only return the first instance of Runner.

        """
        for _, (script, _) in enumerate(self.combined_tickets_generator()):
            if self.debug:
                if kwargs:
                    print(f"{' KWARGS ':#^80}")
                    print(kwargs)
                    _, _, annotations = Runner.runner_arguments()
                    runner_kwargs = Submitter.runner_kwargs_from_script(shlex.split(script)[2:])
                    runner_kwargs.update(kwargs)
                    script = Submitter.script_from_runner_kwargs(annotations, runner_kwargs)
                    script = f"python3 {self.run_toymc} " + " ".join(
                        map(shlex.quote, script.split(" "))
                    )
                    print("\n\n" + f"{' SCRIPT ':#^80}")
                    print(script)
                else:
                    print(f"{' SCRIPT ':#^80}")
                    print(script)
                runner = self.initialized_runner(script)
                # print all parameters
                print("\n\n" + f"{' PARAMETERS ':#^80}")
                print(runner.model.parameters)
                # print all expectation values
                print("\n\n" + f"{' NOMINAL EXPECTATION VALUES ':#^80}")
                try:
                    expectation_values = runner.model.nominal_expectation_values
                    max_key_length = max([len(k) for k in expectation_values.keys()])
                    for k, v in expectation_values.items():
                        print(f"{k:<{max_key_length}}   {v}")
                except NotImplementedError as msg:
                    warnings.warn(str(msg))
                return runner
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
        confidence_levels: List[float] = [0.6827, 0.8, 0.9, 0.95],
    ):
        """Read the likelihood ratio from the output files and calculate the Neyman threshold. The
        threshold will be saved into a json file. The threshold will be sorted based on the elements
        of poi.

        Args:
            free_name: the name of the free hypothesis
            true_name: the name of the true hypothesis
            confidence_levels: the confidence levels to calculate the threshold
                0.6827 = stats.norm.cdf(1) - stats.norm.cdf(-1)

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
            nominal_values = deepcopy(runner_args["nominal_values"])
            generate_values = deepcopy(runner_args["generate_values"])
            needed_kwargs = {
                **nominal_values,
                **generate_values,
            }

            # get the output filename pattern
            output_filename_pattern = search_filename_pattern(
                runner_args["output_filename"].format(**needed_kwargs)
            )

            # read metadata including generate_values
            metadata = self._read_metadata(output_filename_pattern)

            # combine the generate_values from
            # the metadata(where the _rate_multiplier is already calculated)
            needed_kwargs = {**metadata["generate_values"], **needed_kwargs}

            # read poi and poi_expectation
            poi_value, poi_expectation = self._read_poi(metadata, **needed_kwargs)

            # read the likelihood ratio
            results = toyfiles_to_numpy(output_filename_pattern)
            llfree = results[free_name]["ll"]
            lltrue = results[true_name]["ll"]
            llrs = 2.0 * (llfree - lltrue)
            if llrs.min() < 0.0:
                mean_valid = (
                    results[free_name]["valid_fit"] & results[true_name]["valid_fit"]
                ).mean()
                self.logging.warning(
                    f"The lowest log likelihood ratio is negative {llrs.min():.02e}, "
                    f"total fraction of negative log likelihood ratio is "
                    f"{(llrs < 0.0).sum() / len(llrs):.02f}, "
                    f"total fraction of invalid fit is {1 - mean_valid:.02f}, "
                    f"the median of negative log likelihood ratios "
                    f"is {np.median(llrs[llrs < 0.0]):.02e}, "
                    f"there might be a problem in your fitting."
                )
            if len(llrs) < 1000:
                self.logging.warning(
                    "The number of toys is less than 1000, the threshold might not be accurate!"
                )

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
                    threshold[threshold_key] = deepcopy(threshold_value)
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

    def _read_metadata(self, output_filename_pattern):
        """Read metadata from the output files."""
        output_filename_list = sorted(glob(output_filename_pattern))
        metadata_list = get_metadata(output_filename_pattern)
        for m in metadata_list:
            m.pop("date", None)
        if len(set([deterministic_hash(m) for m in metadata_list])) != 1:
            raise ValueError(
                f"The metadata are not the same for all the {len(output_filename_list)} output!"
            )
        metadata = metadata_list[0]
        if metadata["poi"] != self.poi:
            raise ValueError(
                f"The poi in the metadata {metadata['poi']} is not "
                f"the same as the poi {self.poi}!"
            )
        return metadata

    def _read_poi(self, metadata, **kwargs):
        """Read poi and poi_expectation from the metadata, and check if the poi_expectation is
        consistent with the poi_expectation from the model."""
        poi_expectation = kwargs.pop("poi_expectation", None)
        poi_value = kwargs.get(self.poi, None)
        if poi_value is None:
            raise ValueError("Can not find the poi value in the generate_values in metadata!")
        # read expectation_values from metadata
        expectation_values = metadata["expectation_values"]
        # check if the poi_expectation is in expectation_values
        source = self.poi.replace("_rate_multiplier", "")
        if source not in expectation_values:
            warnings.warn(
                f"poi {self.poi} does not corresponds to any source in the model!"
                " so can not calculate the poi_expectation!"
            )
        else:
            _poi_expectation = expectation_values[source]
            if poi_expectation is not None:
                # check if the poi_expectation is consistent
                if not np.isclose(poi_expectation, _poi_expectation):
                    raise ValueError(
                        f"The poi_expectation from model {poi_expectation} is not "
                        f"the same as the poi_expectation from toymc {_poi_expectation}!"
                    )
            else:
                warnings.warn(
                    "Can not find the poi_expectation in the generate_values, "
                    "so will not check if the poi_expectation "
                    "are consistent!"
                )
                poi_expectation = _poi_expectation
        return poi_value, poi_expectation

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
            _threshold = [
                i["threshold"]
                for i in inputs
                if i["hashed_keys"]["generate_values"] == dict(zip(names, p))
            ]
            if len(_threshold) > 1:
                raise ValueError(
                    f"More than one threshold for nominal_values {nominal_values} "
                    f"confidence_level {confidence_level}, and generate_values keys {names}!"
                )
            values[tuple(indices)] = _threshold[0]

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
            if threshold_key in threshold:
                # if threshold_key in threshold, get the poi_values and threshold_values directly
                poi_values = threshold[threshold_key][poi]
                threshold_values = threshold[threshold_key]["threshold"]
            else:
                # if threshold_key not in threshold,
                # interpolate the threshold from the existing threshold
                if len(hashed_keys["generate_values"]) == 0:
                    warnings.warn(
                        f"If hypothesis {hypothesis} does not contain any values except "
                        f"poi, the nominal values should be in limit_threshold, so that "
                        f"the threshold can be interpolated from the existing threshold."
                    )
                    func_list.append(None)
                    continue

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
