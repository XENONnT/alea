from copy import deepcopy
from itertools import product
import logging

import numpy as np
import scipy.stats as sps
import multihist as mh
from blueice.likelihood import BinnedLogLikelihood, UnbinnedLogLikelihood

logging.basicConfig(level=logging.INFO)


class BlueiceDataGenerator:
    """A class for generating data from a blueice likelihood term.

    Attributes:
        ll: The blueice likelihood term.
        binned (bool): True if the likelihood term is binned.
        bincs (list): The bin centers of the likelihood term.
        direction_names (list): The names of the directions of the likelihood term.
        source_histograms (list): The histograms of the sources of the likelihood term.
        data_lengths (list): The number of bins of each component of the likelihood term.
        dtype (list): The data type of the likelihood term.
        last_kwargs (dict): The last kwargs used to generate data.
        mus: The expected number of events of each source of the likelihood term.
        parameters (list): The parameters of the likelihood term.
        ll_term (BinnedLogLikelihood or UnbinnedLogLikelihood): A blueice likelihood term.

    """

    def __init__(self, ll_term):
        """Initialize the BlueiceDataGenerator."""
        if isinstance(ll_term, BinnedLogLikelihood):
            self.binned = True
        elif isinstance(ll_term, UnbinnedLogLikelihood):
            self.binned = False
        else:
            raise NotImplementedError

        ll = deepcopy(ll_term)
        bincs = []  # bin centers of each component
        direction_names = []
        dtype = []
        data_lengths = []  # list of number of bins of each component
        analysis_space = ll_term.base_model.config["analysis_space"]

        source_histograms = {}
        for n in ll.source_name_list:
            source_histograms[n] = mh.Histdd(dimensions=analysis_space)

        for name, bin_edges in analysis_space:
            bincs.append(0.5 * (bin_edges[1:] + bin_edges[:-1]))
            dtype.append((name, float))
            data_lengths.append(len(bin_edges) - 1)
            direction_names.append(name)
        # IDEA: make source a string, not an int
        dtype.append(("source", int))

        data_binc = np.zeros(np.prod(data_lengths), dtype=dtype)
        for i, l in enumerate(product(*bincs)):
            for n, v in zip(direction_names, l):
                data_binc[n][i] = v

        ll.set_data(data_binc)

        self.ll = ll
        self.direction_names = direction_names
        self.source_histograms = source_histograms
        self.data_lengths = data_lengths
        self.dtype = dtype
        self.last_kwargs = {}
        self.first_call = True
        self.mus = ll.base_model.expected_events()
        self.parameters = list(ll.shape_parameters.keys())
        self.parameters += [n + "_rate_multiplier" for n in ll.rate_parameters.keys()]

    def simulate(self, filter_kwargs=True, n_toys=None, sample_n_toys=False, **kwargs):
        """Simulate toys for each source.

        Args:
            filter_kwargs (bool, optional (default=True)): If True,
                only parameters of the ll object are accepted as kwargs. Defaults to True.
            n_toys (int, optional (default=None)): If not None,
                a fixed number n_toys of toys is generated for each source component.
                Defaults to None.
            sample_n_toys (bool, optional (default=False)): If True,
                the number of toys is sampled from a Poisson distribution with mean n_toys.
                Defaults to False. Only works if n_toys is not None.

        Keyword Args:
            kwargs: The parameters pasted to the likelihood function.

        Returns:
            numpy.array: Array of simulated data for all sources in the given analysis space.
            The index "source" indicates the corresponding source of an entry.
            The dtype follows self.dtype.

        """
        self.compute_pdfs_and_mus(filter_kwargs=filter_kwargs, **kwargs)

        if n_toys is not None:
            if sample_n_toys:
                self.n_toys_rv = sps.poisson(n_toys).rvs()
                n_sources = np.full(len(self.ll.base_model.sources), self.n_toys_rv)
            else:
                n_sources = np.full(len(self.ll.base_model.sources), n_toys)
        else:
            # Generate a number n_source (according to the expectation value)
            # of toys for each source component:
            n_sources = sps.poisson(self.mus).rvs()
            if len(self.ll.base_model.sources) == 1:
                n_sources = np.array([n_sources])

        r_data = np.zeros(np.sum(n_sources), dtype=self.dtype)
        i_write = 0
        for i, n_source in enumerate(n_sources):
            if n_source > 0:  # dont generate if 0
                source_name = self.ll.source_name_list[i]
                rvs = self.source_histograms[source_name].get_random(n_source)
                for j, n in enumerate(self.direction_names):
                    r_data[n][i_write : i_write + n_source] = rvs[:, j]
                r_data["source"][i_write : i_write + n_source] = i
                i_write += n_source
        return r_data

    def compute_pdfs_and_mus(self, filter_kwargs=True, **kwargs) -> None:
        """Compute PDFs of the sources for the given parameters.

        Args:
            filter_kwargs (bool, optional (default=True)): If True,
                only parameters of the ll object are accepted as kwargs. Defaults to True.
            kwargs: The parameters pasted to the likelihood function.

        """
        if filter_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in self.parameters + ["livetime_days"]}

        # check if the cached generator may be used:
        unmatched_item = set(self.last_kwargs.items()) ^ set(kwargs.items())
        kwargs_changed = len(unmatched_item) != 0
        if self.first_call or kwargs_changed:
            ret = self.ll(full_output=True, **kwargs)
            if isinstance(ret, float):
                logging.warning("ERROR, generator kwarg outside range?")
                logging.warning(kwargs)
            _, mus, ps_array = ret
            for n, p in zip(self.ll.source_name_list, ps_array):
                self.source_histograms[n].histogram = p.reshape(self.data_lengths)
                if not self.binned:
                    self.source_histograms[n] *= self.source_histograms[n].bin_volumes()
            self.mus = mus
            self.last_kwargs = kwargs
            self.first_call = False
