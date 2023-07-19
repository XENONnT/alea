import os
from pydoc import locate

import numpy as np
import alea


def get_analysis_space(analysis_space: dict) -> list:
    eval_analysis_space = []

    for element in analysis_space:
        for key, value in element.items():
            if isinstance(value, str) and value.startswith("np."):
                eval_element = (key, eval(value))
            elif isinstance(value, str):
                eval_element = (key,
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
        template_folder_list (list): list of possible base folders where
            templates are located. If a folder starts with alea/,
            the alea folder is used as base.
            Ordered by priority.

    Returns:
        dict: adapted likelihood config
    """
    template_folder = None
    for template_folder in template_folder_list:
        # if template folder starts with alea: get location of alea
        if template_folder.startswith("alea/"):
            alea_dir = os.path.dirname(os.path.abspath(alea.__file__))
            template_folder = os.path.join(alea_dir, template_folder.replace("alea/", ""))
        # check if template folder exists
        if not os.path.isdir(template_folder):
            template_folder = None
        else:
            break
    # raise error if no template folder is found
    if template_folder is None:
        raise FileNotFoundError("No template folder found. Please provide a valid template folder.")

    likelihood_config["analysis_space"] = get_analysis_space(
        likelihood_config["analysis_space"])

    likelihood_config["default_source_class"] = locate(
        likelihood_config["default_source_class"])

    for source in likelihood_config["sources"]:
        source["templatename"] = os.path.join(
            template_folder, source["template_filename"])
    return likelihood_config
