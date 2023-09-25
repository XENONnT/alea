:orphan:

Statistical models
==================

The statistical model contains the likelihood, model parameters and data. It can generate toy data, fit the model to (toy)data and compute confidence intervals.
The basic functionality is defined in the :class:`alea.model.StatisticalModel` class. You can inherit the base functionality and add your own likelihood and data generation function.

A very simple statistical model that can be used to understand the concept is the :class:`alea.examples.gaussian_model.GaussianModel`. It has a Gaussian likelihood and Gaussian data generation function. The model parameters are the mean and the standard deviation of the Gaussian distribution.

The more complex :class:`alea.models.blueice_extended_model.BlueiceExtendedModel` model is based on `blueice <https://github.com/JelleAalbers/blueice>`_ and can handle template-based likelihoods. For more details on the `BlueIceExtendedModel`, please refer to the :doc:`/blueice_extended_model/structure` page.
