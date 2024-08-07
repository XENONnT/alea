import os
import re
import json
import yaml
import importlib_resources
import itertools
import blueice
from glob import glob
from copy import deepcopy
from pydoc import locate
import logging
from hashlib import sha256
from base64 import b32encode
from collections.abc import Mapping
from typing import Any, List, Dict, Tuple, Optional, Union, cast, get_args, get_origin
from blueice.pdf_morphers import Morpher
from itertools import product

import h5py
import matplotlib.pyplot as plt

# These imports are needed to evaluate strings
import numpy  # noqa: F401
import numpy as np  # noqa: F401
from scipy import stats  # noqa: F401
from scipy.stats import chi2

logging.basicConfig(level=logging.INFO)


MAX_FLOAT = np.sqrt(np.finfo(np.float32).max)


class ReadOnlyDict:
    """A read-only dict."""

    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def __repr__(self):
        return repr(self._data)

    def get(self, key, default=None):
        return self._data.get(key, default)

    # Prevent changes
    def __setitem__(self, key, value):
        raise TypeError(
            "This dictionary is read-only, please initialize a new one in order to change it."
        )

    def __delitem__(self, key):
        raise TypeError(
            "This dictionary is read-only, please initialize a new one in order to change it."
        )

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


def evaluate_numpy_scipy_expression(value: str):
    """Evaluate numpy(np) and scipy.stats expression."""
    if value.startswith("stats."):
        return eval(value)
    elif value.startswith("np.") or value.startswith("numpy."):
        return eval(value)
    else:
        raise ValueError(f"Expression {value} not understood.")


def evaluate_numpy_scipy_expression_in_dict(d: dict):
    """Evaluate numpy(np) and scipy.stats expression in a dict.

    Example:
        >>> evaluate_numpy_scipy_expression_in_dict({'a': 'np.arange(0, 2, 1)', 'b': [0, 1]})
        {'a': [0, 1], 'b': [0, 1]}

    """
    d_copy = deepcopy(d)
    for k, v in d_copy.items():
        if isinstance(v, str):
            d_copy[k] = evaluate_numpy_scipy_expression(v).tolist()
    return d_copy


def get_analysis_space(analysis_space: list) -> list:
    """Convert analysis_space to a list of tuples with evaluated values."""
    eval_analysis_space = []

    for element in analysis_space:
        for key, value in element.items():
            if isinstance(value, str) and value.startswith("np."):
                eval_element = (key, evaluate_numpy_scipy_expression(value))
            elif isinstance(value, str):
                if "," in value:
                    eval_element = (key, np.fromstring(value, dtype=float, sep=","))
                else:
                    eval_element = (key, np.fromstring(value, dtype=float, sep=" "))
            elif isinstance(value, list):
                eval_element = (key, np.array(value))
            else:
                raise ValueError(f"analysis_space for dimension {key} not understood.")

            eval_analysis_space.append(eval_element)
    return eval_analysis_space


def _prefix_file_path(
    config: dict, template_folder_list: list, ignore_keys: List[str] = ["name", "histname"]
):
    """Prefix file path with template_folder_list whenever possible.

    Args:
        config (dict): dictionary contains file path
        template_folder_list (list): list of possible base folders. Ordered by priority.
        ignore_keys (list, optional (default=["name", "histname"])):
        keys to be ignored when prefixing

    """
    for key in config.keys():
        if isinstance(config[key], str) and key not in ignore_keys:
            try:
                config[key] = get_file_path(config[key], template_folder_list)
            except RuntimeError:
                pass


def adapt_likelihood_config_for_blueice(
    likelihood_config: dict, template_folder_list: list
) -> dict:
    """Adapt likelihood config to be compatible with blueice.

    Args:
        likelihood_config (dict): likelihood config dict
        template_folder_list (list): list of possible base folders. Ordered by priority.

    Returns:
        dict: adapted likelihood config

    """

    likelihood_config_copy = deepcopy(likelihood_config)

    likelihood_config_copy["analysis_space"] = get_analysis_space(
        likelihood_config_copy["analysis_space"]
    )

    _prefix_file_path(likelihood_config_copy, template_folder_list)

    if "default_source_class" in likelihood_config_copy:
        default_source_class = locate(likelihood_config_copy["default_source_class"])
        if default_source_class is None:
            raise ValueError(f"Could not find {likelihood_config_copy['default_source_class']}!")
        likelihood_config_copy["default_source_class"] = default_source_class

    for source in likelihood_config_copy["sources"]:
        if "template_filename" in source:
            source["templatename"] = get_file_path(
                source["template_filename"], template_folder_list
            )
        if "class" in source:
            source_class = locate(source["class"])
            if source_class is None:
                raise ValueError(f"Could not find {source['class']}!")
            source["class"] = source_class
        if "template_filenames" in source:
            source["templatenames"] = [
                get_file_path(template_filename, template_folder_list)
                for template_filename in source["template_filenames"]
            ]
        if source.get("efficiency_name", None):
            source["apply_efficiency"] = True
        _prefix_file_path(source, template_folder_list)
    return likelihood_config_copy


def load_yaml(file_name: str):
    """Load data from yaml file."""
    with open(get_file_path(file_name), "r") as file:
        data = yaml.safe_load(file)
    return data


def load_json(file_name: str):
    """Load data from json file."""
    with open(get_file_path(file_name), "r") as file:
        data = json.load(file)
    return data


def _get_abspath(file_name):
    """Get the abspath of the file.

    Raise FileNotFoundError when not found in any subfolder

    """
    for sub_dir in ("examples/configs", "examples/templates"):
        p = os.path.join(_package_path(sub_dir), file_name)
        if glob(formatted_to_asterisked(p)):
            return p
    raise FileNotFoundError(f"Cannot find {file_name}")


def _package_path(sub_directory):
    """Get the abs path of the requested sub folder."""
    return importlib_resources.files("alea") / sub_directory


def formatted_to_asterisked(formatted, wildcards: Optional[Union[str, List[str]]] = None):
    """Convert formatted string to asterisk Sometimes a parameter(usually shape parameter) is not
    specified in formatted string, this function replace the parameter with asterisk.

    Args:
        formatted (str): formatted string
        wildcards (str or list, optional (default=None)):
            wildcards to be replaced with asterisk.

    Returns:
        str: asterisked string

    Examples:
        >>> formatted_to_asterisked("a_{a:.2f}_b_{b:d}")
        "a_*_b_*"
        >>> formatted_to_asterisked("a_{a:.2f}_b_{b:d}", wildcards="a")
        "a_*_b_{b:d}"

    """
    # find all wildcards if wildcards is None
    if wildcards is None:
        return re.sub("{(.*?)}", "*", formatted)

    # convert wildcards to list if wildcards is a string
    if isinstance(wildcards, str):
        wildcards = [wildcards]
    else:
        if not isinstance(wildcards, list):
            raise ValueError(
                f"wildcards must be a string or list of strings, not {type(wildcards)}"
            )

    # replace wildcards with asterisk
    asterisked = formatted
    for found in re.findall("{(.*?)}", formatted):
        for wildcard in wildcards:
            if wildcard in found:
                asterisked = asterisked.replace("{" + found + "}", "*")
    return asterisked


def get_file_path(fname, folder_list: Optional[List[str]] = None):
    """Find the full path to the resource file Try 5 methods in the following order.

    #. fname begin with '/', return absolute path
    #. folder begin with '/', return folder + name
    #. can get file from _get_abspath, return alea internal file path
    #. can be found in local installed ntauxfiles, return ntauxfiles absolute path
    #. can be downloaded from MongoDB, download and return cached path

    Args:
        fname (str): file name
        folder_list (list, optional (default=None)):
            list of possible base folders. Ordered by priority.
            The function will search for file from the first folder in the list,
            and return the first found file immediately without searching the rest folders.

    Returns:
        str: full path to the resource file

    """
    if os.path.exists(fname):
        return fname

    if folder_list is None:
        folder_list = []
    # 1. From absolute path
    # Usually Config.default is a absolute path
    if fname.startswith("/"):
        return fname

    # 2. From local folder
    # Use folder as prefix
    for folder in folder_list:
        if folder.startswith("/"):
            fpath = os.path.join(folder, fname)
            if glob(formatted_to_asterisked(fpath)):
                logging.info(f"Load {fname} successfully from {fpath}")
                return fpath

    # 3. From alea internal files
    try:
        return _get_abspath(fname)
    except FileNotFoundError:
        pass

    # raise error when can not find corresponding file
    raise RuntimeError(f"Can not find {fname}, please check your file system")


def get_template_folder_list(likelihood_config, extra_template_path: Optional[str] = None):
    """Get a list of template_folder from likelihood_config."""
    if "template_folder" not in likelihood_config:
        # return empty list if template_folder is not specified
        likelihood_config["template_folder"] = []
    if isinstance(likelihood_config["template_folder"], str):
        template_folder_list = [likelihood_config["template_folder"]]
    elif isinstance(likelihood_config["template_folder"], list):
        template_folder_list = likelihood_config["template_folder"]
    elif likelihood_config["template_folder"] is None:
        template_folder_list = []
    else:
        raise ValueError("template_folder must be either a string or a list of strings.")
    # Add extra_template_path to the end of template_folder_list
    if extra_template_path is not None:
        template_folder_list.append(extra_template_path)
    return list(set(template_folder_list))


def asymptotic_critical_value(
    confidence_interval_kind: str, confidence_level: float, degree_of_freedom: Optional[int] = None
):
    """Return the critical value for the confidence interval.

    Args:
        confidence_interval_kind (str): confidence interval kind, either 'lower', 'upper' or
            'central'
        confidence_level (float): confidence level
        degree_of_freedom (int, optional (default=None)): degree of freedom

    Returns:
        float: critical value

    Raises:
        ValueError: if confidence_interval_kind is not 'lower', 'upper' or 'central'
        ValueError: if degree_of_freedom is not None and not 1, when confidence_interval_kind is
            'lower' or 'upper'

    """
    if confidence_interval_kind in {"lower", "upper"}:
        if (degree_of_freedom is not None) and (degree_of_freedom != 1):
            raise ValueError(
                f"degree_of_freedom must be 1 for {confidence_interval_kind} confidence interval"
            )
        critical_value = chi2(1).isf(2 * (1.0 - confidence_level))
    elif confidence_interval_kind == "central":
        if degree_of_freedom is None:
            critical_value = chi2(1).isf(1.0 - confidence_level)
        else:
            critical_value = chi2(degree_of_freedom).isf(1.0 - confidence_level)
    else:
        raise ValueError(
            f"confidence_interval_kind must be either 'lower', 'upper' or 'central', "
            f"not {confidence_interval_kind}"
        )
    return critical_value


def within_limits(value, limits):
    """Returns True if value is within limits."""
    if limits is None:
        return True
    elif limits[0] is None:
        return value <= limits[1]
    elif limits[1] is None:
        return value >= limits[0]
    else:
        return limits[0] <= value <= limits[1]


def clip_limits(value) -> Tuple[float, float]:
    """Clip limits to be within [-MAX_FLOAT, MAX_FLOAT] by converting None to -MAX_FLOAT and
    MAX_FLOAT."""
    if value is None:
        value = [-MAX_FLOAT, MAX_FLOAT]
    else:
        if value[0] is None:
            value[0] = -MAX_FLOAT
        if value[1] is None:
            value[1] = MAX_FLOAT
    return value


def can_assign_to_typing(value_type, target_type) -> bool:
    """Check if value_type can be assigned to target_type. This is useful when converting Runner's
    argument into strings.

    Args:
        value_type: type of the value, might be float, int, etc.
        target_type: type of the target, might be Optinal, Union, etc.

    """
    if get_origin(target_type) is Union:
        # If the target type is a Union (like Optional)
        return any(can_assign_to_typing(value_type, t) for t in get_args(target_type))
    else:
        if get_origin(target_type):
            return issubclass(value_type, get_origin(target_type))
        else:
            return issubclass(value_type, target_type)


def add_i_batch(filename: str) -> str:
    """Add i_batch to filename."""
    if "i_batch" in filename:
        raise ValueError("i_batch already in filename")
    fpat_split = os.path.splitext(filename)
    return fpat_split[0] + "_{i_batch:d}" + fpat_split[1]


def search_filename_pattern(filename: str) -> str:
    """Return pattern for a given existing filename. This is needed because sometimes the filename
    is not appended by "_{i_batch:d}". We need to distinguish between the two cases and return the
    correct pattern.

    Returns:
        str: existing pattern for filename, either filename or filename w/ inserted "_*"

    """
    # try to add a * to the filename to read all the files
    fpat_split = os.path.splitext(filename)
    _filename = fpat_split[0] + "_*" + fpat_split[1]
    if len(sorted(glob(_filename))) != 0:
        pattern = _filename
    else:
        pattern = filename
    filename_list = sorted(glob(pattern))
    if len(filename_list) == 0:
        raise ValueError(f"Can not find any output file {filename}!")
    return pattern


def get_metadata(output_filename_pattern: str) -> list:
    """Get metadata from output files."""
    output_filename_list = sorted(glob(output_filename_pattern))
    metadata_list = []
    for _output_filename in output_filename_list:
        with h5py.File(_output_filename, "r", libver="latest", swmr=True) as ipt:
            metadata = dict(
                zip(
                    ipt.attrs.keys(),
                    [json.loads(ipt.attrs[key]) for key in ipt.attrs.keys()],
                )
            )
        metadata_list.append(metadata)
    return metadata_list


def can_expand_grid(variations: dict) -> bool:
    """Check if variations can be expanded into a grid.

    Example:
        >>> can_expand_grid({'a': [1, 2], 'b': [3, 4]})
        True

    """

    # check if all values are lists or no values is list
    is_list = [isinstance(value, list) for value in variations.values()]
    if {True, False}.issubset(is_list):
        raise ValueError(
            "If some values in variations are lists, "
            "all values must be lists or no values is list. "
            f"But you are mixing lists and non-lists in {variations}."
        )
    if all(is_list):
        return True
    else:
        return False


def expand_grid_dict(variations: List[Union[dict, str]]) -> List[Union[dict, str]]:
    """Expand dict into a list of dict, according to the itertools.product method, if necessary.

    Args:
        variations (list): variations to be expanded

    Example:
        >>> expand_grid_dict(["free", {"a": 1, "b": 3}, {"a": 'np.arange(1, 3)', "b": [3, 4]}])
        [
            "free",
            {"a": 1, "b": 3},
            {"a": 1, "b": 3},
            {"a": 1, "b": 4},
            {"a": 2, "b": 3},
            {"a": 2, "b": 4},
        ]

    """

    result = cast(List[Union[dict, str]], [])
    for v in variations:
        # convert str to list first
        if isinstance(v, dict):
            v = evaluate_numpy_scipy_expression_in_dict(v)
        # expand to grid if necessary
        if isinstance(v, dict) and can_expand_grid(v):
            result += convert_to_vary(v)
        else:
            result.append(v)
    return result


def convert_variations(variations: dict, iteration) -> list:
    """Convert variations to a list of dict, according to the iteration method.

    Args:
        variations (dict): variations to be converted
        iteration: iteration method, either zip or itertools.product

    Returns:
        list: a list of dict

    """

    # evaluate numpy and scipy expression in variations
    variations = evaluate_numpy_scipy_expression_in_dict(variations)

    # expand to grid if necessary
    for k, v in variations.items():
        if not isinstance(v, list):
            raise ValueError(f"variations {k} must be a list, not {v} with {type(v)}")
        variations[k] = expand_grid_dict(v)
    result = [dict(zip(variations, deepcopy(t))) for t in iteration(*variations.values())]
    if result:
        return result
    else:
        return [{}]


def convert_to_zip(to_zip: Dict[str, List]) -> List[Dict[str, Any]]:
    """Convert dict into a list of dict, according to the zip method.

    Example:
        >>> convert_to_zip({'a': [1, 2], 'b': [3, 4]})
        [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]

    """
    return convert_variations(to_zip, zip)


def convert_to_vary(to_vary: Dict[str, List]) -> List[Dict[str, Any]]:
    """Convert dict into a list of dict, according to the itertools.product method.

    Example:
        >>> convert_to_vary({'a': [1, 2], 'b': [3, 4]})
        [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]

    """
    return convert_variations(to_vary, itertools.product)


def convert_to_in_common(in_common: Dict[str, Any]) -> Dict[str, Any]:
    """Expand the values in in_common, according to the itertools.product method, if necessary. This
    usually happens to the hypotheses.

    Example:
        >>> convert_to_in_common({'hypotheses': ['free', {'a': [1, 2], 'b': [3, 4]}]})
        {
            "hypotheses": [
                "free",
                {"a": 1, "b": 3},
                {"a": 1, "b": 4},
                {"a": 2, "b": 3},
                {"a": 2, "b": 4},
            ]
        }

    """
    for k, v in in_common.items():
        if isinstance(v, list) and (k != "hypotheses") and (k != "confidence_levels"):
            raise ValueError(
                f"except hypotheses, in_common can not contain list, "
                f"you might need to put {(k, v)} in to_zip or to_vary"
            )
    if "hypotheses" in in_common:
        in_common["hypotheses"] = expand_grid_dict(in_common["hypotheses"])
    return in_common


def compute_variations(to_zip, to_vary, in_common) -> list:
    """Compute variations of Runner from to_zip, to_vary and in_common. By priority, the order is
    to_zip, to_vary, in_common. The values in to_zip will overwrite the keys in to_vary and
    in_common. The values in to_vary will overwrite the keys in in_common.

    Args:
        to_zip (dict): variations to be zipped
        to_vary (dict): variations to be varied
        in_common (dict): variations in common

    Returns:
        list: a list of dict

    """
    zipped = convert_to_zip(to_zip=to_zip)
    varied = convert_to_vary(to_vary=to_vary)

    combined = [
        {**convert_to_in_common(in_common), **v, **z} for z, v in itertools.product(zipped, varied)
    ]
    return combined


def make_hashable(obj):
    """Convert a container hierarchy into one that can be hashed.

    See http://stackoverflow.com/questions/985294

    """
    if isinstance(obj, Mapping):
        # Convert immutabledict etc for json decoding
        obj = dict(obj)
    try:
        hash(obj)
    except TypeError:
        if isinstance(obj, dict):
            return tuple((k, make_hashable(v)) for (k, v) in sorted(obj.items()))
        elif isinstance(obj, np.ndarray):
            return tuple(obj.tolist())
        elif hasattr(obj, "__iter__"):
            return tuple(make_hashable(o) for o in obj)
        else:
            raise TypeError("Can't make_hashable object of type %r" % type(obj))
    else:
        return obj


def deterministic_hash(thing, length=10):
    """Return a base32 lowercase string of length determined from hashing a container hierarchy.

    Edited from strax: strax/utils.py

    """
    hashable = make_hashable(thing)
    jsonned = json.dumps(hashable)
    # disable bandit
    digest = sha256(jsonned.encode("ascii")).digest()
    return b32encode(digest)[:length].decode("ascii").lower()


def signal_multiplier_estimator(
    signal: np.ndarray,
    background: np.ndarray,
    data: np.ndarray,
    iteration=100,
    diagnostic=False,
) -> float:
    """Estimate the best-fit signal multiplier using perturbation theory. The method tries to solve
    the critial point of the likelihood function by perturbation theory, where the likelihood
    function is defined as the binned Poisson likelihood function, given signal, background models
    and data.

    Args:
        signal (np.ndarray): signal model
        background (np.ndarray): background model
        data (np.ndarray): data array
        iteration (int, optional (default=100)): number of iterations
    Returns:
        float: best-fit signal multiplier

    """
    mask = (signal > 0) | (background > 0)
    if np.any(data[~mask] > 0):
        raise ValueError("Data has non-zero values where signal and background is zero.")

    sig = signal[mask].ravel()
    bkg = background[mask].ravel()
    obs = data[mask].ravel()

    @np.errstate(invalid="ignore", divide="ignore")
    def correction_on_multiplier(x):
        exp = sig * x + bkg
        return np.sum(np.where(exp > 0, (obs / exp - 1) * sig, 0)) / np.sum(
            np.where(exp > 0, obs * sig**2 / exp**2, 0)
        )

    # For underfluctutation case, the best-fit multiplier could be negative
    # in which case the perturbation theory may not converge or be negative.
    # Thus we clip it to be non-negative.
    x = np.sum(obs - bkg) / np.sum(sig)
    xs = [x]
    for _ in range(iteration):
        x += correction_on_multiplier(x)
        x = np.clip(x, 0, None)
        xs.append(x)
    if diagnostic:
        plt.plot(xs, marker=".")
        plt.xlabel("Iteration")
        plt.ylabel("x")
    return x


class IndexMorpher(Morpher):
    """IndexMorpher is a morpher which applies no interpolation."""

    def get_anchor_points(self, bounds, n_models=None):
        grid = [par.keys() for _, (par, _, _) in self.shape_parameters.items()]
        return list(product(*grid))

    def make_interpolator(self, f, extra_dims, anchor_models):
        return lambda z: f(anchor_models[tuple(z)])


blueice.pdf_morphers.MORPHERS["IndexMorpher"] = IndexMorpher
