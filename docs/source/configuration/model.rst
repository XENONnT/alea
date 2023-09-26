:orphan:

Configuration of statistical models
===================================

**UNDER CONSTRUCTION**

The :doc:`BlueIceExtendedModel</blueice_extended_model/structure>` can be fully initialized using a YAML configuration file. In the we'll walk you through the basic structure and how to set up your own model.

The file consists of two parts: The definition of all parameters of the model (``parameter_definition``) and the likelihood configuration (``likelihood_config``).
This means, the basic structure will look something like this:

.. code-block:: yaml

    parameter_definition:
        parameter_1:
            ...
        parameter_2:
            ...
        ...
    likelihood_config:
        ...

The ``parameter_definition`` section 