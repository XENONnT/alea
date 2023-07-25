import os
import yaml
import pkg_resources
from pydoc import locate
from warnings import warn

import numpy as np


def get_analysis_space(analysis_space: dict) -> list:
    eval_analysis_space = []

    for element in analysis_space:
        for key, value in element.items():
            if value.startswith("np."):
                eval_element = (key, eval(value))
            else:
                eval_element = (
                    key,
                    np.fromstring(
                        value,
                        dtype=float,
                        sep=" "))
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

    likelihood_config["analysis_space"] = get_analysis_space(
        likelihood_config["analysis_space"])

    likelihood_config["default_source_class"] = locate(
        likelihood_config["default_source_class"])

    for source in likelihood_config["sources"]:
        source["templatename"] = get_file_path(
            source["template_filename"], template_folder_list)
    return likelihood_config


def load_yaml(file_name: str):
    """Load data from yaml file."""
    with open(get_file_path(file_name), 'r') as file:
        data = yaml.safe_load(file)
    return data


def _get_abspath(file_name):
    """Get the abspath of the file. Raise FileNotFoundError when not found in any subfolder"""
    for sub_dir in ('model_configs', 'runner_configs', 'templates'):
        p = os.path.join(_package_path(sub_dir), file_name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'Cannot find {file_name}')


def _package_path(sub_directory):
    """Get the abs path of the requested sub folder"""
    return pkg_resources.resource_filename('alea', f'{sub_directory}')


def get_file_path(fname, folder_list=None):
    """Find the full path to the resource file
    Try 5 methods in the following order

    #. fname begin with '/', return absolute path
    #. url_base begin with '/', return url_base + name
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
    # Use url_base as prefix
    for folder in folder_list:
        if folder.startswith('/'):
            fpath = os.path.join(folder, fname)
            if os.path.exists(fpath):
                warn(f'Load {fname} successfully from {fpath}')
                return fpath

    # 3. From alea internal files
    try:
        return _get_abspath(fname)
    except FileNotFoundError:
        pass

    # raise error when can not find corresponding file
    raise RuntimeError(f'Can not find {fname}, please check your file system')
