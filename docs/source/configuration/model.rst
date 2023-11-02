:orphan:

Configuration of statistical models
===================================
The :doc:`BlueIceExtendedModel</blueice_extended_model/structure>` can be fully initialized using a YAML configuration file. In the following, we'll walk you through the basic structure and how to set up your own model.

The file consists of two parts: The definition of all parameters of the model (``parameter_definition``) and the likelihood configuration (``likelihood_config``).
This means, the basic structure will look something like this:

.. code-block:: yaml

    parameter_definition:
      parameter_name_1:
        property_1: value_1
        property_2: value_2
        # ... more properties
      parameter_name_2:
        # ... definition missing
      # ... more parameters
    likelihood_config:
      template_folder: null
      likelihood_terms:
      - name: term_1
        # ... some technical things missing here
        sources:
        - name: source_1
          # ... some things missing here
          parameters:
            - parameter_name_1
          template_filename: template_1.h5
        - name: source_2
          # ... source definition missing
        # ... more sources
      # ... more likelihood terms

Let's have a closer look at the two parts.

Parameter definition
--------------------

The ``parameter_definition`` section is a dictionary of all parameters of the model. Each parameter is defined by a name and a dictionary of properties. You can specify any property, which is an attribute in the :class:`Parameter<alea.parameters.Parameter>` class. One exception is the ``name`` attribute, which is determined by the key in the dictionary.
An overview with some examples is provided in the table below.

.. list-table:: Property Definitions
   :widths: 15 20 50
   :header-rows: 1
   :class: tight-table

   * - Property Name
     - Format
     - Description
   * - ``nominal_value``
     - float, optional
     - The nominal value of the parameter.
   * - ``fittable``
     - bool, optional
     - Indicates if the parameter is fittable or always fixed.
   * - ``ptype``
     - str, optional
     - The ptype of the parameter.
   * - ``uncertainty``
     - float or str, optional
     - The uncertainty of the parameter. If a string, it can be evaluated as a numpy or scipy function to define non-gaussian constraints.
   * - ``relative_uncertainty``
     - bool, optional
     - Indicates if the uncertainty is relative to the nominal_value.
   * - ``blueice_anchors``
     - list, optional
     - Anchors for blueice template morphing. Blueice will load the template for the provided values and then interpolate for any value in between.
   * - ``fit_limits``
     - Tuple[float, float], optional
     - The limits for fitting the parameter.
   * - ``parameter_interval_bounds``
     - Tuple[float, float], optional
     - Limits for computing confidence intervals.
   * - ``fit_guess``
     - float, optional
     - The initial guess for fitting the parameter.
   * - ``description``
     - str, optional
     - A description of the parameter. This is a long description that may span multiple lines in the table cell.


For the BlueIceExtendedModel, the ``ptype`` can be used to optimize performance. The following options are available:

  * ``'rate'``: The parameter is a rate parameter. This means it linearly scales the expectation value of a source in the model.
  * ``'livetime'``: The parameter is a livetime parameter. Similarly to a rate parameter it linearly scales expectation values, however, it does so for the entire likelihood term it is connected to. Also, in contrast to a rate multiplier, it is almost never a fittable parameter.
  * ``'shape'``: The parameter is a shape parameter. For blueice likelihoods this means that it is a template morphing parameter. For this, a template is provided at at least two ``blueice_anchor`` values and the template is interpolated for any value in between.
  * ``'efficiency'``: The parameter is an efficiency parameter. This means that it scales the expectation value of a source in the model for a given livetime and rate parameter.


Likelihood configuration
------------------------

The likelihood configuration defines individual terms of a summed likelihood.
Each term is defined as a list of sources (background and signal).
Importantly, the parameters are linked to the sources and likelihood terms here.
Also, the template files are specified.

In addition to the likelihood terms in the list, an **ancillary likelihood term** will be added automatically to the summed likelihood.
It contains all the constraint terms of nuisance parameters, which are defined via the ``uncertainty`` property in the parameter definition.
