:orphan:

Structure of BlueiceExtendedModel
=================================

The :class:`alea.models.blueice_extended_model.BlueiceExtendedModel` is a flexible statistical model built on top of the `blueice <https://github.com/JelleAalbers/blueice>`_ likelihood framework. It allows you to construct binned or unbinned likelihoods with template morphing, nuisance parameters, and sophisticated data generation.

Overview
========

The BlueiceExtendedModel combines:

* **Parameter definitions** — specification of all model parameters (rates, shapes, efficiencies, etc.)
* **Likelihood configuration** — definition of likelihood terms, sources, and their linking to parameters
* **Data generators** — automatic generation of toy data from blueice likelihoods
* **Fitting and inference** — built-in support for parameter fitting, confidence intervals, and sensitivity studies

Key Attributes
===============

* ``parameters`` (:class:`alea.parameters.Parameters`): Collection of all model parameters with nominal values, uncertainties, and constraints
* ``data`` (dict or list): Datasets for each likelihood term (can be set from structured arrays or loaded from files)
* ``is_data_set`` (bool): Whether data has been provided to the model
* ``likelihood_names`` (list): Names of all likelihood terms including "ancillary" (for constraints)
* ``livetime_parameter_names`` (list): Optional livetime parameter for each likelihood term
* ``data_generators`` (list): :class:`alea.simulators.BlueiceDataGenerator` instances for toy data generation

Configuration
==============

The BlueiceExtendedModel is typically initialized from a YAML configuration file using :meth:`BlueiceExtendedModel.from_config`, which defines two sections:

1. **parameter_definition** — all parameters and their properties
2. **likelihood_config** — likelihood terms, sources, and their connection to templates and parameters

For comprehensive details on the configuration structure, see :doc:`/configuration/model`.

Example Usage
=============

Initialize from a configuration file:

.. code-block:: python

    from alea.models.blueice_extended_model import BlueiceExtendedModel

    model = BlueiceExtendedModel.from_config(
        "model_config.yaml",
        template_path="/path/to/templates"
    )

Generate toy data and fit:

.. code-block:: python

    # Generate toy data
    toy_data = model.generate_data()

    # Set the data
    model.data = toy_data

    # Fit the model
    fit_result = model.fit()
    print(f"Best-fit POI: {fit_result['poi_name']}")

    # Compute confidence interval
    ci = model.compute_confidence_interval(
        "poi_name",
        confidence_level=0.9
    )
    print(f"90% CL interval: [{ci[0]:.3f}, {ci[1]:.3f}]")

Template Morphing
=================

For shape parameters marked with ``ptype: 'shape'`` in the parameter definition, the model performs template morphing: it loads templates at anchor points and interpolates for intermediate parameter values. This is controlled by the ``blueice_anchors`` property in the parameter definition.

For more details, see the parameter definition section of :doc:`/configuration/model`.

See Also
========

* :doc:`/configuration/model` — comprehensive guide to model and likelihood configuration
* :doc:`/configuration/runner` — guide to runner and submitter configuration
* :class:`alea.model.StatisticalModel` — base class with common inference methods
* `blueice documentation <https://github.com/JelleAalbers/blueice>`_ — likelihood framework
