from copy import deepcopy
from itertools import product
import logging

import numpy as np
import scipy.stats as sps
import blueice
import multihist as mh

logging.basicConfig(level=logging.INFO)


class BlueiceDataGenerator:
    """
    A class for generating data from a blueice likelihood term.

    Args:
        ll_term (blueice.likelihood.BinnedLogLikelihood
            or blueice.likelihood.UnbinnedLogLikelihood):
            A blueice likelihood term.
    """

    def __init__(self, ll_term):
        if isinstance(ll_term, blueice.likelihood.BinnedLogLikelihood):
            binned = True
        elif isinstance(ll_term, blueice.likelihood.UnbinnedLogLikelihood):
            binned = False
        else:
            raise NotImplementedError

        ll = deepcopy(ll_term)
        bins = []  # bin edges
        bincs = []  # bin centers of each component
        direction_names = []
        dtype = []
        data_length = 1  # number of bins in nD histogram
        data_lengths = []  # list of number of bins of each component
        for direction in ll.base_model.config['analysis_space']:
            bins.append(direction[1])
            binc = 0.5 * (direction[1][1::] + direction[1][0:-1])
            bincs.append(binc)
            dtype.append((direction[0], float))
            data_length *= len(direction[1]) - 1
            data_lengths.append(len(direction[1]) - 1)
            direction_names.append(direction[0])
        dtype.append(('source', int))
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
        logging.debug("initing simulator, binned: " + str(binned))
        self.binned = binned
        self.bincs = bincs
        self.direction_names = direction_names
        self.source_histograms = source_histograms
        self.data_lengths = data_lengths
        self.dtype = dtype
        self.last_kwargs = {"FAKE_PARAMETER": None}
        self.mus = ll.base_model.expected_events()
        self.parameters = list(ll.shape_parameters.keys())
        self.parameters += [
            n + '_rate_multiplier' for n in ll.rate_parameters.keys()
        ]

    def simulate(self,
                 filter_kwargs=True,
                 n_toys=None,
                 sample_n_toys=False,
                 **kwargs):
        """simulate toys for each source

        Args:
            filter_kwargs (bool, optional): If True, only parameters of
                the ll object are accepted as kwargs. Defaults to True.
            n_toys (int, optional): If not None: a fixed number n_toys of
                toys is generated for each source component.

        Returns:
            structured array: array of simulated data for all sources in
                the given analysis space. The index 'source' indicates
                the corresponding source of an entry.
        """
        if filter_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in self.parameters}

        unmatched_item = set(self.last_kwargs.items()) ^ set(kwargs.items())
        logging.debug("filtered kwargs in simulate: " + str(kwargs))
        logging.debug("unmatched_item in simulate: " + str(unmatched_item))
        if ("FAKE_PARAMETER"
                in self.last_kwargs.keys()) or (len(unmatched_item) != 0):
            ret = self.ll(full_output=True, **kwargs)  # result, mus, ps
            if type(ret) == float:
                logging.warning("ERROR, generator kwarg outside range?")
                logging.warning(kwargs)
            #print("ret,kwargs",ret,kwargs)
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
                n_sources = np.full(len(self.ll.base_model.sources),
                                    self.n_toys_rv)
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
            if 0 < n_source:  #dont generate if 0
                rvs = self.source_histograms[i].get_random(n_source)
                for j, n in enumerate(self.direction_names):
                    r_data[n][i_write:i_write + n_source] = rvs[:, j]
                r_data['source'][i_write:i_write + n_source] = i
                i_write += n_source
        logging.debug("return simulated data with length: " + str(len(r_data)))
        return r_data


class simulate_nearest:
    def __init__(self, ll):
        self.ll = ll
        self.parameters = list(ll.shape_parameters.keys())
        self.parameters += [
            n + '_rate_multiplier' for n in ll.rate_parameters.keys()
        ]

    def simulate(self, **kwargs):
        call_args_shape = {}
        shape_coordinates = {}
        rate_multipliers = {}
        for k, v in kwargs.items():
            if k in self.parameters:
                if k.endswith('_rate_multiplier'):
                    rate_multipliers[k.replace('_rate_multiplier', '')] = v
                else:
                    call_args_shape[k] = v
        if len(self.ll.shape_parameters) == 0:
            return self.ll.base_model.simulate(
                rate_multipliers=rate_multipliers)

        for sp in self.ll.shape_parameters.keys():
            if self.ll.shape_parameters[sp][2] is None:
                #Non-numerical parameters have their default values in shape_paramters, otherwise in config
                shape_coordinates[sp] = self.ll.base_model.config[sp]
            else:
                shape_coordinates[sp] = self.ll.shape_parameters[sp][2]
        call_args_closest = {}
        for sp in call_args_shape:
            #store key not value (same for numerical, but makes a difference for string nuisance pars)
            diff_dir = {
                abs(k - call_args_shape[sp]): k
                for k, val in self.ll.shape_parameters[sp][0].items()
            }
            call_args_closest[sp] = diff_dir[min(diff_dir.keys())]
        shape_coordinates.update(call_args_closest)
        shape_key = []
        for sp in self.ll.shape_parameters.keys():
            shape_key.append(shape_coordinates[sp])
        shape_key = tuple(shape_key)
        m = self.ll.anchor_models[shape_key]

        return m.simulate(rate_multipliers=rate_multipliers)
