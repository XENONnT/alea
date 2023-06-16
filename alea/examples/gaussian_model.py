from alea.statistical_model import StatisticalModel
import scipy.stats as stats
import numpy as np
from iminuit import Minuit
from iminuit.util import make_func_code


class GaussianModel(StatisticalModel):
    def __init__(self, nominal_mu, nominal_sigma):
        """
        Initialise a model of a gaussian measurement (hatmu),
        where the model has parameters mu and sigma
        For illustration, we show how required nominal parameters can be added to the init
        sigma is fixed in this example.
        """
        self.nominal_values = {"mu": nominal_mu,
                               "sigma": nominal_sigma}
        super().__init__(fixed_parameters={"sigma": nominal_sigma},
                         )

    def ll(self, mu=None, sigma=None):
        if mu is None:
            mu = self.nominal_values.get("mu", None)
        if sigma is None:
            sigma = self.nominal_values.get("sigma", None)

        hat_mu = self.get_data()[0]['hat_mu'][0]
        return stats.norm.logpdf(x=hat_mu, loc=mu, scale=sigma)

    def generate_data(self, mu=None, sigma=None):
        if mu is None:
            mu = self.nominal_values.get("mu", None)
        if sigma is None:
            sigma = self.nominal_values.get("sigma", None)

        hat_mu = stats.norm(loc=mu, scale=sigma).rvs()
        data = [np.array([(hat_mu,)], dtype=[('hat_mu', float)])]
        return data

