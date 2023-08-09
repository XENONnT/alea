from copy import deepcopy
from itertools import product
import logging

import numpy as np
import scipy.stats as sps
import multihist as mh
from blueice.likelihood import BinnedLogLikelihood, UnbinnedLogLikelihood

logging.basicConfig(level=logging.INFO)


class BlueiceDataGenerator:
    """
    A class for generating data from a blueice likelihood term.

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
        """Initialize the BlueiceDataGenerator"""
        if isinstance(ll_term, BinnedLogLikelihood):
            self.binned = True
        elif isinstance(ll_term, UnbinnedLogLikelihood):
            self.binned = False
        else:
            raise NotImplementedError
        logging.debug("initing simulator, binned: " + str(self.binned))

        ll = deepcopy(ll_term)
        bins = []  # bin edges
        bincs = []  # bin centers of each component
        direction_names = []
        dtype = []
        data_length = 1  # number of bins in nD histogram
        data_lengths = []  # list of number of bins of each component
        for direction in ll.base_model.config["analysis_space"]:
            bins.append(direction[1])
            binc = 0.5 * (direction[1][1::] + direction[1][0:-1])
            bincs.append(binc)
            dtype.append((direction[0], float))
            data_length *= len(direction[1]) - 1
            data_lengths.append(len(direction[1]) - 1)
            direction_names.append(direction[0])
        dtype.append(("source", int))
        logging.debug("init simulate_interpolated with bins: " + str(bins))

        data_binc = np.zeros(data_length, dtype=dtype)
        for i, l in enumerate(product(*bincs)):
            for n, v in zip(direction_names, l):
                data_binc[n][i] = v

        ll.set_data(data_binc)
        source_histograms = []
        for i in range(len(ll.base_model.sources)):
            source_histograms.append(mh.Histdd(bins=bins))

        self.ll = ll
        self.bincs = bincs
        self.direction_names = direction_names
        self.source_histograms = source_histograms
        self.data_lengths = data_lengths
        self.dtype = dtype
        self.last_kwargs = {"FAKE_PARAMETER": None}
        self.mus = ll.base_model.expected_events()
        self.parameters = list(ll.shape_parameters.keys())
        self.parameters += [
            n + "_rate_multiplier" for n in ll.rate_parameters.keys()
        ]

    def simulate(
            self,
            filter_kwargs=True,
            n_toys=None,
            sample_n_toys=False,
            **kwargs):
        """simulate toys for each source

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
        if filter_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in self.parameters + ["livetime_days"]}

        unmatched_item = set(self.last_kwargs.items()) ^ set(kwargs.items())
        logging.debug("filtered kwargs in simulate: " + str(kwargs))
        logging.debug("unmatched_item in simulate: " + str(unmatched_item))
        # check if the cached generator may be used:
        if ("FAKE_PARAMETER"
                in self.last_kwargs.keys()) or (len(unmatched_item) != 0):
            ret = self.ll(full_output=True, **kwargs)  # result, mus, ps
            if type(ret) == float:
                logging.warning("ERROR, generator kwarg outside range?")
                logging.warning(kwargs)
            _, mus, ps_array = ret
            for i in range(len(self.ll.base_model.sources)):
                self.source_histograms[i].histogram = ps_array[i].reshape(
                    self.data_lengths)
                if not self.binned:
                    logging.debug(f"Source {str(self.ll.base_model.sources[i].name)} is not binned. Multiplying histogram with bin volumes.")
                    self.source_histograms[i] *= self.source_histograms[
                        i].bin_volumes()
                    logging.debug("n after multiplying: " + str(self.source_histograms[i].n))
            self.mus = mus
            self.last_kwargs = kwargs
            logging.debug(f"mus of simulate: " + str(mus))

        if n_toys is not None:
            if sample_n_toys:
                self.n_toys_rv = sps.poisson(n_toys).rvs()
                n_sources = np.full(
                    len(self.ll.base_model.sources), self.n_toys_rv)
            else:
                n_sources = np.full(len(self.ll.base_model.sources), n_toys)
        else:
            # Generate a number n_source (according to the expectation value)
            # of toys for each source component:
            n_sources = sps.poisson(self.mus).rvs()
            logging.debug("number of events drawn from Poisson: " + str(n_sources))
            if len(self.ll.base_model.sources) == 1:
                n_sources = np.array([n_sources])

        r_data = np.zeros(np.sum(n_sources), dtype=self.dtype)
        i_write = 0
        for i, n_source in enumerate(n_sources):
            if n_source > 0:  # dont generate if 0
                rvs = self.source_histograms[i].get_random(n_source)
                for j, n in enumerate(self.direction_names):
                    r_data[n][i_write:i_write + n_source] = rvs[:, j]
                r_data["source"][i_write:i_write + n_source] = i
                i_write += n_source
        logging.debug("return simulated data with length: " + str(len(r_data)))
        return r_data
