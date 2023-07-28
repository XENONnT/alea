import os
import re
import yaml
import pkg_resources
from glob import glob
from copy import deepcopy
from pydoc import locate
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)


def get_analysis_space(analysis_space: dict) -> list:
    eval_analysis_space = []

    for element in analysis_space:
        for key, value in element.items():
            if isinstance(value, str) and value.startswith("np."):
                eval_element = (key, eval(value))
            elif isinstance(value, str):
                eval_element = (
                    key,
                    np.fromstring(value,
                                  dtype=float,
                                  sep=" "))
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
        template_folder_list (list): list of possible base folders.
            Ordered by priority.

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
    return pkg_resources.resource_filename('alea', f'{sub_directory}')


def formatted_to_asterisked(formatted):
    """
    Convert formatted string to asterisk
    Sometimes a parameter(usually shape parameter) is not specified in formatted string,
    this function replace the parameter with asterisk.
    """
    asterisked = formatted
    for found in re.findall("\{(.*?)\}", formatted):
        asterisked = asterisked.replace('{' + found + '}', "*")
    return asterisked


def get_file_path(fname, folder_list=None):
    """Find the full path to the resource file
    Try 5 methods in the following order

    #. fname begin with '/', return absolute path
    #. folder begin with '/', return folder + name
    #. can get file from _get_abspath, return alea internal file path
    #. can be found in local installed ntauxfiles, return ntauxfiles absolute path
    #. can be downloaded from MongoDB, download and return cached path
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
