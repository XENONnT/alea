import os
import re
import yaml
import importlib_resources
from glob import glob
from copy import deepcopy
from pydoc import locate
import logging
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO)


MAX_FLOAT = np.sqrt(np.finfo(np.float32).max)


def get_analysis_space(analysis_space: dict) -> list:
    """Convert analysis_space to a list of tuples with evaluated values."""
    eval_analysis_space = []

    for element in analysis_space:
        for key, value in element.items():
            if isinstance(value, str) and value.startswith("np."):
                eval_element = (key, eval(value))
            elif isinstance(value, str):
                eval_element = (
                    key,
                    np.fromstring(value, dtype=float, sep=" "))
            elif isinstance(value, list):
                eval_element = (key, np.array(value))
            else:
                raise ValueError(f"analysis_space for dimension {key} not understood.")

            eval_analysis_space.append(eval_element)
    return eval_analysis_space


def adapt_likelihood_config_for_blueice(
    likelihood_config: dict,
    template_folder_list: list) -> dict:
    """
    Adapt likelihood config to be compatible with blueice.

    Args:
        likelihood_config (dict): likelihood config dict
        template_folder_list (list): list of possible base folders. Ordered by priority.

    Returns:
        dict: adapted likelihood config
    """

    likelihood_config_copy = deepcopy(likelihood_config)

    likelihood_config_copy["analysis_space"] = get_analysis_space(
        likelihood_config_copy["analysis_space"])

    likelihood_config_copy["default_source_class"] = locate(
        likelihood_config_copy["default_source_class"])

    for source in likelihood_config_copy["sources"]:
        source["templatename"] = get_file_path(
            source["template_filename"], template_folder_list)
    return likelihood_config_copy


def load_yaml(file_name: str):
    """Load data from yaml file."""
    with open(get_file_path(file_name), 'r') as file:
        data = yaml.safe_load(file)
    return data


def _get_abspath(file_name):
    """Get the abspath of the file. Raise FileNotFoundError when not found in any subfolder"""
    for sub_dir in ('examples/configs', 'examples/templates'):
        p = os.path.join(_package_path(sub_dir), file_name)
        if glob(formatted_to_asterisked(p)):
            return p
    raise FileNotFoundError(f'Cannot find {file_name}')


def _package_path(sub_directory):
    """Get the abs path of the requested sub folder"""
    return importlib_resources.files('alea') / sub_directory


def formatted_to_asterisked(formatted, wildcards: Optional[str or list]=None):
    """
    Convert formatted string to asterisk
    Sometimes a parameter(usually shape parameter) is not specified in formatted string,
    this function replace the parameter with asterisk.

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
        wildcards = re.findall("\{(.*?)\}", formatted)
    else:
        if isinstance(wildcards, str):
            wildcards = [wildcards]
        else:
            if not isinstance(wildcards, list):
                raise ValueError(
                    f"wildcards must be a string or list of strings, not {type(wildcards)}")
        _wildcards = []
        for wildcard in wildcards:
            _wildcards += re.findall("\{" + wildcard + "(.*?)\}", formatted)
        wildcards = _wildcards
    # replace wildcards with asterisk
    for wildcard in wildcards:
        asterisked = asterisked.replace('{' + wildcard + '}', "*")
    return asterisked


def get_file_path(fname, folder_list=None):
    """Find the full path to the resource file
    Try 5 methods in the following order

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
    if fname.startswith('/'):
        return fname

    # 2. From local folder
    # Use folder as prefix
    for folder in folder_list:
        if folder.startswith('/'):
            fpath = os.path.join(folder, fname)
            if glob(formatted_to_asterisked(fpath)):
                logging.info(f'Load {fname} successfully from {fpath}')
                return fpath

    # 3. From alea internal files
    try:
        return _get_abspath(fname)
    except FileNotFoundError:
        pass

    # raise error when can not find corresponding file
    raise RuntimeError(f'Can not find {fname}, please check your file system')


def get_template_folder_list(likelihood_config):
    """Get a list of template_folder from likelihood_config"""
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
        raise ValueError(
            "template_folder must be either a string or a list of strings.")
    return template_folder_list


def within_limits(value, limits):
    """Returns True if value is within limits"""
    if limits is None:
        return True
    elif limits[0] is None:
        return value <= limits[1]
    elif limits[1] is None:
        return value >= limits[0]
    else:
        return limits[0] <= value <= limits[1]


def clip_limits(value):
    """
    Clip limits to be within [-MAX_FLOAT, MAX_FLOAT]
    by converting None to -MAX_FLOAT and MAX_FLOAT.
    """
    if value is None:
        value = [-MAX_FLOAT, MAX_FLOAT]
    else:
        if value[0] is None:
            value[0] = -MAX_FLOAT
        if value[1] is None:
            value[1] = MAX_FLOAT
    return value


def add_i_batch(filename):
    if 'i_batch' in filename:
        raise ValueError('i_batch already in filename')
    fpat_split = os.path.splitext(filename)
    return fpat_split[0] + '_{i_batch:d}' + fpat_split[1]


import copy
import json
import itertools

from tqdm import tqdm
import mergedeep


def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1, 2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def variations_sanity_check(
        parameters_to_vary,
        parameters_to_zip,
        parameters_in_common):
    """Check that the signal_rate_multiplier is not varied when signal_expectation is not None"""
    ptv_op = 'output_file' not in json.dumps(parameters_to_vary)
    ptz_op = 'output_file' not in json.dumps(parameters_to_zip)
    assert ptv_op and ptz_op, 'output_file should only be provided in parameters_in_common'
    ptz_srm = parameters_to_zip.get('generate_args', [{}])[0].get('signal_rate_multiplier', None)
    ptv_srm = parameters_to_vary.get('generate_args', [{}])[0].get('signal_rate_multiplier', None)
    ptz_se = parameters_to_zip.get('signal_expectation', None)
    ptv_se = parameters_to_vary.get('signal_expectation', None)
    pic_se = parameters_in_common.get('signal_expectation', None)
    srm_flag = (ptz_srm is None and ptv_srm is None)
    se_flag = (ptz_se is None and ptv_se is None and pic_se is None)
    assert srm_flag or se_flag, 'signal_rate_multiplier cannot be varied when signal_expectation is not None'


def compute_parameters_to_zip(parameters_to_zip, silent=False):
    """Compute all variations of parameters_to_zip"""
    # 0. if nothing to zip, return empty list
    if len(parameters_to_zip) == 0:
        return [{}]

    ptz = copy.deepcopy(parameters_to_zip)
    # 1. get all lists
    all_lists = []
    for key, value in ptz.items():
        if isinstance(value, list):
            all_lists.append(value)
        elif isinstance(value, dict):
            if len(value) == 1:
                key_inner, item = list(value.keys())[0], list(value.values())[0]
                if isinstance(item, list):
                    all_lists.append(item)
                    ptz[key] = [{key_inner: list_value} for list_value in item]
                else:
                    raise NotImplementedError(
                        'parameters_to_zip not implemented for dict with values of type('
                        + str(type(value)) + ')')
            else:
                ptz[key] = list(dict_product(value))
                all_lists.append(ptz[key])
        else:
            raise NotImplementedError(
                'parameters_to_zip not implemented for type(' + type(value) + ')')

    # 2. check that all values have the same length
    if len(all_lists) > 0:
        it = iter(all_lists)
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            raise ValueError('not all lists have same length!')

    # 3. put all values in a list of dicts
    varied_dicts_zip = []
    for values in zip(*ptz.values()):
        this_dict = {key: value for key, value in zip(ptz.keys(), values)}
        varied_dicts_zip.append(this_dict)

    # 4. zipping sanity check
    if len(all_lists) > 0:
        # TODO: be strict about this
        if len(all_lists[0]) != len(varied_dicts_zip):
            raise Exception(
                'Zipping failed. You probably escaped checking with a special case.'
            )
    else:
        if not silent:
            print(
                'Cannot check sanity of zip - better provide a list like, var: [1, 2, 3]'
            )

    return varied_dicts_zip


def compute_parameters_to_vary(parameters_to_vary) -> list:
    """Compute all variations of parameters_to_vary using itertools.product"""
    # 0. if nothing to vary, return empty list
    if len(parameters_to_vary) == 0:
        return [{}]

    for key, value in parameters_to_vary.items():
        if isinstance(value, str) and value.startswith('np.'):
            parameters_to_vary[key] = eval(value).tolist()

    # 1. allows variations inside of dicts
    # make dict_product of all dicts in parameters_to_vary
    for k in copy.deepcopy(parameters_to_vary):
        if isinstance(parameters_to_vary[k], dict):
            parameters_to_vary[k] = [
                item for item in dict_product(parameters_to_vary[k])
            ]

    # 2. these are the variations of parameters_to_vary
    cartesian_product = itertools.product(*parameters_to_vary.values())
    parameter_names = parameters_to_vary.keys()

    variations_to_return = []
    for variation in cartesian_product:
        variations_to_return.append(dict(zip(parameter_names, variation)))

    return variations_to_return


def compute_variations(
        parameters_to_zip,
        parameters_to_vary,
        parameters_in_common,
        special_parameters=None,
        silent=False):
    """
    if parameters are defined in multiple places the order or precedence is(high to low):
    1. parameters_to_zip
    2. parameters_to_vary
    3. parameters_in_common
    """
    varied_dicts = compute_parameters_to_vary(parameters_to_vary=parameters_to_vary)
    zipped_dicts = compute_parameters_to_zip(parameters_to_zip=parameters_to_zip, silent=silent)

    combined_variations = list(itertools.product(varied_dicts, zipped_dicts))

    if special_parameters is None:
        special_parameters = [
            'generate_args', 'livetime'
        ]

    if len(combined_variations) > 0:
        keys_are_shared = bool(
            set(combined_variations[0][0]) & set(combined_variations[0][1]))
        if keys_are_shared:
            shared_parameters = list(
                set(combined_variations[0][0]).intersection(combined_variations[0][1]))
            if not silent:
                print(f'There are shared keys between parameters: {shared_parameters}.')
            problematic_parameters = []
            for parameter in shared_parameters:
                if parameter not in special_parameters:
                    problematic_parameters.append(parameter)

            if len(problematic_parameters) == 0:
                if not silent:
                    print(
                        'Did not find big problems.'
                        + ' But you still need to watch out that everything is correct but only special_parameters are shared.'
                    )
            else:
                if len(problematic_parameters) > 1:
                    message = ' '.join(problematic_parameters) + ' are shared.'
                else:
                    message = ' '.join(problematic_parameters) + ' is shared.'
                raise Exception(message)

    # TODO: make sure that hypotheses only exist in parameters_in_common
    # similar restrictions should also apply to other parameters
    hypotheses = []
    for extra_arg in parameters_in_common['hypotheses']:
        if type(extra_arg) is dict:
            if all([type(v) is list for k, v in extra_arg.items()]):
                hypotheses += list(dict_product(extra_arg))
            elif all([not type(v) is list for k, v in extra_arg.items()]):
                pass
            else:
                raise Exception(
                    'The hypotheses dict should either contain only lists or only non-lists.'
                )
        else:
            hypotheses += [extra_arg]
    parameters_in_common['hypotheses'] = hypotheses

    merged_combinations = []
    for variation, zipped in tqdm(combined_variations, disable=silent):
        pic = copy.deepcopy(parameters_in_common)
        mergedeep.merge(pic, variation, zipped)
        merged_combinations.append(pic)
    else:
        return merged_combinations
