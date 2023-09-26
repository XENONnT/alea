:orphan:

Configuration of statistical models
===================================

**UNDER CONSTRUCTION**

The :doc:`BlueIceExtendedModel</blueice_extended_model/structure>` can be fully initialized using a YAML configuration file. In the we'll walk you through the basic structure and how to set up your own model.

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
   :width: 100
   :widths: 20 15 15 1
   :header-rows: 1

   * - Property Name
     - Format
     - Default Value
     - Description
   * - ``nominal_value``
     - float, optional
     - None
     - The nominal value of the parameter.
   * - ``fittable``
     - bool, optional
     - None
     - Indicates if the parameter is fittable or always fixed.
   * - ``ptype``
     - str, optional
     - None
     - The ptype of the parameter.
   * - ``uncertainty``
     - float or str, optional
     - None
     - The uncertainty of the parameter. If a string, it can be evaluated as a numpy or scipy function to define non-gaussian constraints.
   * - ``relative_uncertainty``
     - bool, optional
     - None
     - Indicates if the uncertainty is relative to the nominal_value.
   * - ``blueice_anchors``
     - list, optional
     - None
     - Anchors for blueice template morphing. Blueice will load the template for the provided values and then interpolate for any value in between.
   * - ``fit_limits``
     - Tuple[float, float], optional
     - None
     - The limits for fitting the parameter.
   * - ``parameter_interval_bounds``
     - Tuple[float, float], optional
     - None
     - Limits for computing confidence intervals.
   * - ``fit_guess``
     - float, optional
     - None
     - The initial guess for fitting the parameter.
   * - ``description``
     - str, optional
     - None
     - A description of the parameter. This is a long description that may span multiple lines in the table cell.




# TODO: write about the special role of livetime parameters and rate multipliers for the BlueIceExtendedModel.