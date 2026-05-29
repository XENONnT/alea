:orphan:

Configuration of runners
========================

The :class:`alea.runner.Runner` orchestrates toy Monte Carlo simulations: it initializes a statistical model, generates or loads toy data, performs fits, and computes confidence intervals. Runners are typically submitted to batch systems via :class:`alea.submitter.Submitter` implementations, but they can also be used standalone for custom workflows.

Basic Configuration Structure
==============================

Runners are configured via a YAML file that specifies the statistical model, fitting options, and output settings. A minimal configuration includes:

.. code-block:: yaml

    statistical_model: alea.models.BlueiceExtendedModel
    statistical_model_config: model_config.yaml
    poi: parameter_name
    n_mc: 100
    hypotheses: ["free"]
    output_filename: results.ii.h5

Statistical Model Configuration
--------------------------------

* ``statistical_model`` (str, required): Fully qualified class name of the statistical model (e.g., ``alea.models.BlueiceExtendedModel``, ``alea.examples.gaussian_model.GaussianModel``)
* ``statistical_model_config`` (str, optional): Path to a YAML file defining the model's parameters and likelihood (see :doc:`/configuration/model`). If provided, ``parameter_definition`` and ``likelihood_config`` arguments must not be included.
* ``poi`` (str, required): Name of the parameter of interest for hypothesis tests and confidence intervals

Fitting and Confidence Intervals
---------------------------------

* ``n_mc`` (int, required): Number of Monte Carlo toys to generate and fit
* ``hypotheses`` (list, optional (default=["free"])): List of hypotheses to test. Each hypothesis can be:

  * A string: ``"free"`` (fit all parameters), ``"zero"`` (POI fixed to 0), or ``"true"`` (POI fixed to nominal value)
  * A dict: ``{"poi_expectation": value}`` or ``{parameter_name: fixed_value}`` to set specific parameter values

* ``compute_confidence_interval`` (bool, optional (default=False)): Whether to compute confidence intervals for the POI
* ``confidence_level`` (float, optional (default=0.9)): Confidence level for intervals (e.g., 0.9 for 90% CL)
* ``confidence_interval_kind`` (str, optional (default="central")): Kind of interval: ``"central"``, ``"upper"``, or ``"lower"``
* ``confidence_interval_root_find`` (str, optional (default="brentq")): Root-finding algorithm: ``"brentq"`` or ``"extremal"``

Toy Data Options
----------------

* ``toydata_mode`` (str, optional (default="generate_and_store")): How to handle toy data:

  * ``"generate"`` — generate fresh toys for each job
  * ``"generate_and_store"`` — generate toys and save to file for reuse
  * ``"read"`` — read toys from an existing file (``toydata_filename`` must be provided)
  * ``"no_toydata"`` — skip toy generation (for fitting real data only)

* ``toydata_filename`` (str, optional (default="toydata.ii.h5")): Path to toy data file (used if ``toydata_mode`` is ``"read"`` or ``"generate_and_store"``)
* ``generate_values`` (dict, optional): Parameter values to use when generating toy data. Can include ``"poi_expectation"`` to automatically compute the POI value from a nominal expectation.
* ``seed`` (int, optional): Random seed for reproducibility

Output Options
---------------

* ``output_filename`` (str, optional (default="output.ii.h5")): Path to output file where fit results will be saved. Can include Python format strings like ``{parameter_name}`` to embed parameter values in the filename.
* ``only_toydata`` (bool, optional (default=False)): If ``True``, only generate and save toy data, skip fitting
* ``metadata`` (dict, optional): Custom metadata to store in the output file

Parameter Values
-----------------

* ``nominal_values`` (dict, optional): Nominal values to assign to model parameters (overrides defaults from parameter definitions)
* ``common_hypothesis`` (dict, optional): Parameter values that apply to all hypotheses
* ``fit_strategy`` (dict, optional): Custom fitting strategy to override the model's default

Advanced Options
-----------------

* ``statistical_model_args`` (dict, optional): Additional keyword arguments passed to the statistical model constructor

Example Configuration
======================

Here is a more complete example from the alea test suite:

.. code-block:: yaml

    statistical_model: alea.models.BlueiceExtendedModel
    statistical_model_config: statistical_model.yaml

    poi: wimp_rate_multiplier

    n_mc: 1000
    hypotheses:
      - free
      - zero
      - {"wimp_rate_multiplier": 15}

    compute_confidence_interval: true
    confidence_level: 0.9
    confidence_interval_kind: central

    generate_values:
      wimp_mass: 50
      poi_expectation: 10

    toydata_mode: generate_and_store
    toydata_filename: toys_wimp_mass_50.ii.h5

    output_filename: results_wimp_mass_50.ii.h5

    nominal_values:
      wimp_mass: 50
      efficiency_factor: 1.0

See Also
--------

For details on configuring the statistical model (parameter definitions and likelihood), refer to :doc:`/configuration/model`.
