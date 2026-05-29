:orphan:

Simulators
==========

Data simulators in alea generate toy data from a statistical model. They allow you to perform Monte Carlo studies by sampling from the likelihood and generating synthetic datasets, which is essential for computing sensitivity studies and conducting hypothesis tests.

The main simulator in alea is the :class:`alea.simulators.BlueiceDataGenerator`, which generates data from blueice likelihood terms used by the :class:`alea.models.blueice_extended_model.BlueiceExtendedModel`. It handles both binned and unbinned likelihoods.

**Key features:**

* Generates toy data from a blueice likelihood term
* Handles both binned and unbinned analysis spaces
* Supports Poisson fluctuations in event counts
* Caches PDF and expected event count computations for efficiency
* Returns structured numpy arrays with source identification

The :class:`alea.simulators.BlueiceDataGenerator` is typically used internally by the :class:`alea.model.StatisticalModel` during toy Monte Carlo studies, but you can also use it directly for custom data generation workflows.
