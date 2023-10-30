.. image:: https://zenodo.org/badge/654100988.svg
   :target: https://zenodo.org/badge/latestdoi/654100988

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/XENONnT/alea/HEAD?labpath=notebooks

.. image:: https://github.com/XENONnT/alea/actions/workflows/pytest.yml/badge.svg?branch=main
   :target: https://github.com/XENONnT/alea/actions/workflows/pytest.yml

.. image:: https://coveralls.io/repos/github/XENONnT/alea/badge.svg?branch=main&kill_cache=1
   :target: https://coveralls.io/github/XENONnT/alea?branch=main&kill_cache=1

.. image:: https://img.shields.io/pypi/v/alea-inference.svg
   :target: https://pypi.python.org/pypi/alea-inference/

.. image:: https://readthedocs.org/projects/alea/badge/?version=latest
   :target: https://alea.readthedocs.io/en/latest/?badge=latest

.. image:: https://www.codefactor.io/repository/github/xenonnt/alea/badge
   :target: https://www.codefactor.io/repository/github/xenonnt/alea

.. image:: https://results.pre-commit.ci/badge/github/XENONnT/alea/main.svg
   :target: https://results.pre-commit.ci/latest/github/XENONnT/alea/main


Alea
====

`alea <https://github.com/XENONnT/alea>`_ is a flexible statistical inference framework. The Python package is designed for constructing, handling, and fitting statistical models, computing confidence intervals and conducting sensitivity studies. It is primarily developed for the `XENONnT dark matter experiment <https://xenonexperiment.org/>`_, but can be used for any statistical inference problem.

If you use alea in your research, please consider citing the software published on `zenodo <https://zenodo.org/badge/latestdoi/654100988>`_.

.. toctree::
    :maxdepth: 1
    :caption: Getting started

    installation.rst
    notebooks/0_introduction.ipynb
    notebooks/1_rate_and_shape.ipynb
    notebooks/2_fitting_and_ci.ipynb
    notebooks/3_sensitivity.ipynb

.. toctree::
    :maxdepth: 1
    :caption: Basic concepts

    basics/parameters.rst
    basics/models.rst
    basics/simulators.rst
    basics/runner.rst
    basics/submitters.rst

.. toctree::
    :maxdepth: 1
    :caption: BlueiceExtendedModel

    blueice_extended_model/structure.rst

.. toctree::
    :maxdepth: 1
    :caption: Statistical model and runner configuration

    configuration/model.rst
    configuration/runner.rst

.. toctree::
    :maxdepth: 2
    :caption: Release notes

    reference/release_notes


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
