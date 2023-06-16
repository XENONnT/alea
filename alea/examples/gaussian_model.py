from alea.statistical_model import StatisticalModel
import scipy.stats as stats
import numpy as np
from iminuit import Minuit
from iminuit.util import make_func_code


class GaussianModel(StatisticalModel):
    def __init__(self, mu, sigma):
        self.nominal_values = {"mu": mu,
                               "sigma": sigma}
        super().__init__(fixed_parameters={"sigma": sigma},
                         )

    def ll(self, mu=None, sigma=None):
        if mu is None:
            mu = self.nominal_values.get("mu", None)
        if sigma is None:
            sigma = self.nominal_values.get("sigma", None)

        hat_mu = self.get_data()['hat_mu'][0]
        return stats.norm.logpdf(x=hat_mu, loc=mu, scale=sigma)

    def generate_data(self, mu=None, sigma=None):
        if mu is None:
            mu = self.nominal_values.get("mu", None)
        if sigma is None:
            sigma = self.nominal_values.get("sigma", None)

        hat_mu = stats.norm(loc=mu, scale=sigma).rvs()
        data = np.array([(hat_mu,)], dtype=[('hat_mu', float)])
        return data

    def make_objective(self, guess=None, bound=None, minus=True, **kwargs):
        if guess is None:
            guess = {}
        if bound is None:
            bound = {}
        names = []
        bounds = []
        guesses = []

        for p in self._parameter_list:
            if p not in kwargs:
                g = guess.get(p, 1)
                b = bound.get(p, None)
                names.append(p)
                guesses.append(g)
                bounds.append(b)

        sign = -1 if minus else 1

        def cost(args):
            # Get the arguments from args, then fill in the ones already fixed in outer kwargs
            call_kwargs = {}
            for i, k in enumerate(names):
                call_kwargs[k] = args[i]
            call_kwargs.update(kwargs)
            return self.ll(**call_kwargs) * sign

        return cost, names, np.array(guesses), bounds

    def fit(self, guess=None, bound=None, verbose=False, **kwargs):
        cost, names, guess, bounds = self.make_objective(minus=True, guess=guess,
                                                         bound=bound, **kwargs)
        minuit_dict = {}
        for i, name in enumerate(names):
            minuit_dict[name] = guess[i]

        class MinuitWrap:
            """Wrapper for functions to be called by Minuit

            s_args must be a list of argument names of function f
            the names in this list must be the same as the keys of
            the dictionary passed to the Minuit call."""

            def __init__(self, f, s_args):
                self.func = f
                self.s_args = s_args
                self.func_code = make_func_code(s_args)

            def __call__(self, *args):
                return self.func(args)

        # Make the Minuit object
        cost.errordef = Minuit.LIKELIHOOD
        m = Minuit(MinuitWrap(cost, names),
                   **minuit_dict)
        for k in self._fixed_parameters:
            m.fixed[k] = True
        for n, b in zip(names, bounds):
            if b is not None:
                print(n, b)
                m.limits[n] = b

        # Call migrad to do the actual minimization
        m.migrad()
        if verbose:
            print(m)
        return m.values.to_dict(), -1 * m.fval
