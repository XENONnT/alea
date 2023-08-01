from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from alea.model import StatisticalModel


class GaussianModel(StatisticalModel):
    """
    A model of a gaussian measurement, where the model has parameters mu and sigma.
    For illustration, we show how required nominal parameters can be added
    to the init sigma is fixed in this example.

    Args:
        parameter_definition (dict or list, optional (default=None)):
            definition of the parameters of the model

    Caution:
        You must define the nominal values of the parameters (mu, sigma)
        in the parameters definition.
    """

    def __init__(
            self,
            parameter_definition: Optional[Dict or List] = None,
            **kwargs,
        ):
        """Initialise a gaussian model."""
        if parameter_definition is None:
            parameter_definition = ["mu", "sigma"]
        super().__init__(parameter_definition=parameter_definition, **kwargs)

    def _ll(self, mu=None, sigma=None):
        """
        Log-likelihood of the model.

        Args:
            mu (float, optional (default=None)): mean of the gaussian,
                if None, the nominal value is used
            sigma (float, optional (default=None)): standard deviation of the gaussian,
                if None, the nominal value is used
        """
        hat_mu = self.data[0]['hat_mu'][0]
        return stats.norm.logpdf(x=hat_mu, loc=mu, scale=sigma)

    def _generate_data(self, mu=None, sigma=None):
        """
        Generate data from the model.

        Args:
            mu (float, optional (default=None)): mean of the gaussian,
                if None, the nominal value is used
            sigma (float, optional (default=None)): standard deviation of the gaussian,
                if None, the nominal value is used

        Returns:
            list: data generated from the model
        """
        hat_mu = stats.norm(loc=mu, scale=sigma).rvs()
        data = [np.array([(hat_mu,)], dtype=[('hat_mu', float)])]
        return data
