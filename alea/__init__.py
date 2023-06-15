import os

_ROOT = os.path.abspath(os.path.dirname(__file__))

from alea import likelihoods

from alea import plotting
from alea import simulators
from alea import template_source
from alea import toymc_running
from alea import utils


def get_data_path(data_path):
    return os.path.join(_ROOT, "data", data_path)
