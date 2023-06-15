import os

_ROOT = os.path.abspath(os.path.dirname(__file__))

from binference import likelihoods

from binference import plotting
from binference import simulators
from binference import template_source
from binference import toymc_running
from binference import utils


def get_data_path(data_path):
    return os.path.join(_ROOT, "data", data_path)
