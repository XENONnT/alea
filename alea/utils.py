import os
import re
import yaml
import importlib_resources
import itertools
from glob import glob
from copy import deepcopy
from pydoc import locate
import logging
from typing import Any, List, Dict, Tuple, Optional, Union, cast, get_args, get_origin

# These imports are needed to evaluate strings
import numpy  # noqa: F401
import numpy as np  # noqa: F401
from scipy import stats  # noqa: F401

logging.basicConfig(level=logging.INFO)


MAX_FLOAT = np.sqrt(np.finfo(np.float32).max)


def evaluate_numpy_scipy_expression(value: str):
    """Evaluate numpy(np) and scipy.stats expression."""
    if value.startswith("stats."):
        return eval(value)
    elif value.startswith("np.") or value.startswith("numpy."):
        return eval(value)
    else:
        raise ValueError(f"Expression {value} not understood.")


def get_analysis_space(analysis_space: dict) -> list:
    """Convert analysis_space to a list of tuples with evaluated values."""
    eval_analysis_space = []

    for element in analysis_space:
        for key, value in element.items():
            if isinstance(value, str) and value.startswith("np."):
                eval_element = (key, evaluate_numpy_scipy_expression(value))
            elif isinstance(value, str):
                eval_element = (key, np.fromstring(value, dtype=float, sep=" "))
            elif isinstance(value, list):
                eval_element = (key, np.array(value))
            else:
                raise ValueError(f"analysis_space for dimension {key} not understood.")

            eval_analysis_space.append(eval_element)
    return eval_analysis_space


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

    likelihood_config_copy["default_source_class"] = locate(
        likelihood_config_copy["default_source_class"]
    )

    for source in likelihood_config_copy["sources"]:
        source["templatename"] = get_file_path(source["template_filename"], template_folder_list)
    return likelihood_config_copy


def load_yaml(file_name: str):
    """Load data from yaml file."""
    with open(get_file_path(file_name), "r") as file:
        data = yaml.safe_load(file)
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

    """
    asterisked = formatted
    # find all wildcards
    if wildcards is None:
        wildcards = re.findall("{(.*?)}", formatted)
    else:
        if isinstance(wildcards, str):
            wildcards = [wildcards]
        else:
            if not isinstance(wildcards, list):
                raise ValueError(
                    f"wildcards must be a string or list of strings, not {type(wildcards)}"
                )
        _wildcards = []
        for wildcard in wildcards:
            _wildcards += re.findall("{" + wildcard + "(.*?)}", formatted)
        wildcards = _wildcards
    # replace wildcards with asterisk
    for wildcard in wildcards:
        asterisked = asterisked.replace("{" + wildcard + "}", "*")
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
    return template_folder_list


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


def expand_grid_dict(variations: List[Union[dict, str]]) -> List[Union[dict, str]]:
    """Expand dict into a list of dict, according to the itertools.product method, if necessary.

    Args:
        variations (list): variations to be expanded

    Example:
        >>> expand_grid_dict(['free', {'a': [1, 2], 'b': [3, 4]}])
        ['free', {'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]

    """

    result = cast(List[Union[dict, str]], [])
    for v in variations:
        if isinstance(v, dict):
            is_list = [isinstance(value, list) for value in v.values()]
            if all(is_list):
                result += convert_to_vary(v)
            if {True, False}.issubset(is_list):
                raise ValueError(
                    "If some values in variations are lists, "
                    "all values must be lists or no values is list. "
                    f"But you are mixing lists and non-lists in {v}."
                )
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
    for k, v in variations.items():
        if isinstance(v, str):
            variations[k] = evaluate_numpy_scipy_expression(v).tolist()
        if not isinstance(variations[k], list):
            raise ValueError(f"variations {k} must be a list, not {v} with {type(v)}")
        variations[k] = expand_grid_dict(variations[k])
    result = [dict(zip(variations, t)) for t in iteration(*variations.values())]
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
        if isinstance(v, list) and (k != "hypotheses"):
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
