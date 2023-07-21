from typing import Optional

from scipy import stats
import numpy as np

from alea.model import StatisticalModel


class GaussianModel(StatisticalModel):
    def __init__(
            self,
            parameter_definition: Optional[dict or list] = None,
            **kwargs,
        ):
        """
        Initialise a model of a gaussian measurement (hatmu),
        where the model has parameters mu and sigma
        For illustration, we show how required nominal parameters can be added to the init
        sigma is fixed in this example.
        """
        if parameter_definition is None:
            parameter_definition = ["mu", "sigma"]
        super().__init__(parameter_definition=parameter_definition, **kwargs)

    def _ll(self, mu=None, sigma=None):
        hat_mu = self.data[0]['hat_mu'][0]
        return stats.norm.logpdf(x=hat_mu, loc=mu, scale=sigma)

    def _generate_data(self, mu=None, sigma=None):
        hat_mu = stats.norm(loc=mu, scale=sigma).rvs()
        data = [np.array([(hat_mu,)], dtype=[('hat_mu', float)])]
        return data
