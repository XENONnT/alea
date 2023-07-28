import json
import warnings
import logging

import blueice
import numpy as np
from scipy.interpolate import interp1d
from inference_interface import template_to_multihist

logging.basicConfig(level=logging.INFO)
can_check_binning = True


class TemplateSource(blueice.HistogramPdfSource):
    """
    A source that constructs a  from a template histogram
    :param templatename: root file to open.
    :param histname: Histogram name.
    :param named_parameters: list of config setting names to pass to .format on histname and filename.
    :param in_events_per_bin:
    if True, histogram is taken to be in events per day / bin,
    if False or absent, taken to be events per day / bin volume
    :param histogram_multiplier: multiply histogram by this number
    :param log10_bins: List of axis numbers.
    If True, bin edges on this axis in the root file are log10() of the actual bin edges.
    """

    def build_histogram(self):
        format_dict = {
            k: self.config[k]
            for k in self.config.get('named_parameters', [])
        }
        logging.debug("loading template")
        templatename = self.config['templatename'].format(**format_dict)
        histname = self.config['histname'].format(**format_dict)

        slice_args = self.config.get("slice_args", {})
        if type(slice_args) is dict:
            slice_args = [slice_args]

        h = template_to_multihist(templatename, histname)

        if self.config.get('normalise_template', False):
            h /= h.n
            if not self.config.get('in_events_per_bin', True):
                h.histogram *= h.bin_volumes()

        for sa in slice_args:
            slice_axis = sa.get("slice_axis", None)
            # Decide if you wish to sum the histogram into lower dimensions or
            sum_axis = sa.get("sum_axis", False)
            collapse_axis = sa.get('collapse_axis', None)
            collapse_slices = sa.get('collapse_slices', None)
            if slice_axis is not None:
                slice_axis_number = h.get_axis_number(slice_axis)
                bes = h.bin_edges[slice_axis_number]
                slice_axis_limits = sa.get(
                    "slice_axis_limits", [bes[0], bes[-1]])
                if sum_axis:
                    logging.debug(f"Slice and sum over axis {slice_axis} from {slice_axis_limits[0]} to {slice_axis_limits[1]}")
                    axis_names = h.axis_names_without(slice_axis)
                    h = h.slicesum(
                        axis=slice_axis,
                        start=slice_axis_limits[0],
                        stop=slice_axis_limits[1])
                    h.axis_names = axis_names
                else:
                    logging.debug(f"Normalization before slicing: {h.n}.")
                    logging.debug(f"Slice over axis {slice_axis} from {slice_axis_limits[0]} to {slice_axis_limits[1]}")
                    h = h.slice(axis=slice_axis,
                                start=slice_axis_limits[0],
                                stop=slice_axis_limits[1])
                    logging.debug(f"Normalization after slicing: {h.n}.")

            if collapse_axis is not None:
                if collapse_slices is None:
                    raise ValueError(
                        "To collapse you must supply collapse_slices")
                h = h.collapse_bins(collapse_slices, axis=collapse_axis)

        self.dtype = []
        for n, _ in self.config['analysis_space']:
            self.dtype.append((n, float))
        self.dtype.append(('source', int))

        # Fix the bin sizes
        if can_check_binning:
            # Deal with people who have log10'd their bins
            for axis_i in self.config.get('log10_bins', []):
                h.bin_edges[axis_i] = 10**h.bin_edges[axis_i]

            # Check if the histogram bin edges are correct
            for axis_i, (_, expected_bin_edges) in enumerate(
                    self.config['analysis_space']):
                expected_bin_edges = np.array(expected_bin_edges)
                seen_bin_edges = h.bin_edges[axis_i]
                # If 1D, hist1d returns bin_edges straight, not as list
                if len(self.config['analysis_space']) == 1:
                    seen_bin_edges = h.bin_edges
                logging.debug("axis_i: " + str(axis_i))
                logging.debug("expected_bin_edges: " + str(expected_bin_edges))
                logging.debug("seen_bin_edges: " + str(seen_bin_edges))
                logging.debug("h.bin_edges type" + str(h.bin_edges))
                if len(seen_bin_edges) != len(expected_bin_edges):
                    raise ValueError(
                        "Axis %d of histogram %s in root file %s has %d bin edges, but expected %d"
                        % (
                            axis_i, histname, templatename, len(seen_bin_edges),
                            len(expected_bin_edges)))
                try:
                    np.testing.assert_almost_equal(
                        seen_bin_edges,
                        expected_bin_edges,
                        decimal=2)
                except AssertionError:
                    warnings.warn(
                        "Axis %d of histogram %s in root file %s has bin edges %s, "
                        "but expected %s. Since length matches, setting it expected values..."
                        % (
                            axis_i, histname, templatename, seen_bin_edges,
                            expected_bin_edges))
                    h.bin_edges[axis_i] = expected_bin_edges

        self._bin_volumes = h.bin_volumes()  # TODO: make alias
        # Shouldn't be in HistogramSource... anyway
        self._n_events_histogram = h.similar_blank_histogram()

        h *= self.config.get('histogram_scale_factor', 1)
        logging.debug(f"Multiplying histogram with histogram_scale_factor {self.config.get('histogram_scale_factor', 1)}. Histogram is now normalised to {h.n}.")

        # Convert h to density...
        if self.config.get('in_events_per_bin'):
            h.histogram /= h.bin_volumes()
        self.events_per_day = (h.histogram * self._bin_volumes).sum()
        logging.debug(f"events_per_day: " + str(self.events_per_day))

        # ... and finally to probability density
        if 0 < self.events_per_day:
            h.histogram /= self.events_per_day

        if self.config.get('convert_to_uniform', False):
            h.histogram = 1./self._bin_volumes

        self._pdf_histogram = h
        logging.debug(f"Setting _pdf_histogram normalised to {h.n}.")

        if np.min(self._pdf_histogram.histogram) < 0:
            raise AssertionError(
                f"There are bins for source {templatename} with negative entries."
                    )

    def simulate(self, n_events):
        dtype = []
        for n, _ in self.config['analysis_space']:
            dtype.append((n, float))
        dtype.append(('source', int))
        ret = np.zeros(n_events, dtype=dtype)
        # t = self._pdf_histogram.get_random(n_events)
        h = self._pdf_histogram * self._bin_volumes
        t = h.get_random(n_events)
        for i, (n, _) in enumerate(self.config.get('analysis_space')):
            ret[n] = t[:, i]
        return ret


class CombinedSource(blueice.HistogramPdfSource):
    """
    Source that inherits structure from TH2DSource by Jelle,
    but takes in lists of histogram names.
    :param weights: weights
    :param histnames: list of filenames containing the histograms
    :param templatenames: list of names of histograms within the hdf5 files
    :param named_parameters : list of names of weights to be applied to histograms.
    Must be 1 shorter than histnames, templatenames
    :param histogram_parameters: names of parameters that should be put in the hdf5/histogram names,
    """

    def build_histogram(self):
        weight_names = self.config.get("weight_names")
        weights = [
            self.config.get(weight_name, 0) for weight_name in weight_names
        ]
        histograms = []
        format_dict = {
            k: self.config[k]
            for k in self.config.get('named_parameters', [])
        }
        histnames = self.config['histnames']
        templatenames = self.config['templatenames']

        slice_args = self.config.get("slice_args", {})
        if type(slice_args) is dict:
            slice_args = [slice_args]
        if type(slice_args[0]) is dict:
            slice_argss = []
            for n in histnames:
                slice_argss.append(slice_args)
        else:
            slice_argss = slice_args

        slice_fractions = []

        for histname, templatename, slice_args in zip(
                histnames, templatenames, slice_argss):
            templatename = templatename.format(**format_dict)
            histname = histname.format(**format_dict)
            h = template_to_multihist(templatename, histname)
            slice_fraction = 1. / h.n
            if h.n == 0.:
                slice_fraction = 0.
            for sa in slice_args:
                slice_axis = sa.get("slice_axis", None)
                # Decide if you wish to sum the histogram into lower dimensions or
                sum_axis = sa.get("sum_axis", False)

                slice_axis_limits = sa.get("slice_axis_limits", [0, 0])
                collapse_axis = sa.get('collapse_axis', None)
                collapse_slices = sa.get('collapse_slices', None)
                if (slice_axis is not None):
                    if sum_axis:
                        axis_names = h.axis_names_without(slice_axis)
                        h = h.slicesum(
                            axis=slice_axis,
                            start=slice_axis_limits[0],
                            stop=slice_axis_limits[1])
                        h.axis_names = axis_names
                    else:
                        h = h.slice(axis=slice_axis,
                                    start=slice_axis_limits[0],
                                    stop=slice_axis_limits[1])

                if collapse_axis is not None:
                    if collapse_slices is None:
                        raise ValueError(
                            "To collapse you must supply collapse_slices")
                    h = h.collapse_bins(collapse_slices, axis=collapse_axis)
            slice_fraction *= h.n
            slice_fractions.append(slice_fraction)

            # Avoid dividing by zero for empty PDFs (if mu=0, does not matter if unnormalised)
            if h.n <= 0.:
                h *= 0.
            else:
                h /= h.n
            histograms.append(h)

        assert len(weights) + 1 == len(histograms)

        # Here a common normalisation is performed (normalisation to bin widths comes later)
        # for histogram in histograms:
        #     histogram/=histogram.n

        for i in range(len(weights)):

            h_comp = histograms[0].similar_blank_histogram()
            h = histograms[i + 1]
            base_part_norm = 0.
            # h_centers = h_comp.bin_centers()
            # h_inds = [range(len(hc)) for hc in h_centers]
            # for inds in product(*h_inds):
            #     bincs = [h_centers[j][k] for j, k in enumerate(inds)]
            #     inds_comp = h_comp.get_bin_indices(bincs)
            #     base_part_norm += histograms[0][inds]
            #     h_comp[inds] = h[inds_comp]
            # print("i", i, "h_comp", h_comp.histogram.shape, h_comp.n)
            # h_comp *= base_part_norm
            hslices = []
            for j, bincs in enumerate(h.bin_centers()):
                hsliced = h_comp.get_axis_bin_index(bincs[0], j)
                hsliceu = h_comp.get_axis_bin_index(bincs[-1], j) + 1
                hslices.append(slice(hsliced, hsliceu))
            base_part_norm = histograms[0][hslices].sum()
            h_comp[hslices] += h.histogram  # TODO check here what norm I want.
            histograms[0] += h_comp * weights[i]

        # Set pdf values that are below 0 to zero:
        histograms[0].histogram[np.isinf(histograms[0].histogram)] = 0.
        histograms[0].histogram[np.isnan(histograms[0].histogram)] = 0.
        histograms[0].histogram[histograms[0].histogram < 0] = 0.

        logging.debug("composite, histograms.n", histograms[0].n)
        if 0 < histograms[0].n:
            h = histograms[0] / histograms[0].n
        else:
            h = histograms[0]
            h.histogram = np.ones(h.histogram.shape)
        self._bin_volumes = h.bin_volumes()  # TODO: make alias
        # Shouldn't be in HistogramSource... anyway
        self._n_events_histogram = h.similar_blank_histogram()

        # Fix the bin sizes
        if can_check_binning:
            # Deal with people who have log10'd their bins
            for axis_i in self.config.get('log10_bins', []):
                h.bin_edges[axis_i] = 10**h.bin_edges[axis_i]

            # Check if the histogram bin edges are correct
            for axis_i, (_, expected_bin_edges) in enumerate(
                    self.config['analysis_space']):
                expected_bin_edges = np.array(expected_bin_edges)
                seen_bin_edges = h.bin_edges[axis_i]
                logging.debug("in axis_i check", axis_i)
                logging.debug("expected", expected_bin_edges)
                logging.debug("seen", seen_bin_edges)
                logging.debug("h.bin_edges type", type(h.bin_edges))
                if len(seen_bin_edges) != len(expected_bin_edges):
                    raise ValueError(
                        "Axis %d of histogram %s in hdf5 file %s has %d bin edges, but expected %d"
                        % (
                            axis_i, histname, templatename, len(seen_bin_edges),
                            len(expected_bin_edges)))
                try:
                    np.testing.assert_almost_equal(
                        seen_bin_edges,
                        expected_bin_edges, decimal=4)
                except AssertionError:
                    warnings.warn(
                        "Axis %d of histogram %s in hdf5 file %s has bin edges %s, "
                        "but expected %s. Since length matches, setting it expected values..."
                        % (
                            axis_i, histname, templatename, seen_bin_edges,
                            expected_bin_edges))
                    h.bin_edges[axis_i] = expected_bin_edges

        h *= self.config.get('histogram_multiplier', 1)
        logging.debug("slice fractions are", slice_fractions)
        h *= slice_fractions[0]

        # Convert h to density...
        logging.debug("hist sum", np.sum(h.histogram))
        if self.config.get('in_events_per_bin'):
            h.histogram /= h.bin_volumes()
        self.events_per_day = (h.histogram * h.bin_volumes()).sum()
        logging.debug("init events per day", self.events_per_day)

        # ... and finally to probability density
        if 0 < self.events_per_day:
            h.histogram /= self.events_per_day
        else:
            h.histogram = np.ones(h.histogram.shape)
        logging.debug("_pdf_histogram sum", h.n)
        self._pdf_histogram = h

    def simulate(self, n_events):
        dtype = []
        for n, _ in self.config['analysis_space']:
            dtype.append((n, float))
        dtype.append(('source', int))
        ret = np.zeros(n_events, dtype=dtype)
        # t = self._pdf_histogram.get_random(n_events)
        h = self._pdf_histogram * self._bin_volumes
        t = h.get_random(n_events)
        for i, (n, _) in enumerate(self.config.get('analysis_space')):
            ret[n] = t[:, i]
        return ret


class SpectrumTemplateSource(blueice.HistogramPdfSource):
    """
    :param spectrum_name: name of bbf json-like spectrum _OR_ function that can be called
    templatename #3D histogram (Etrue,S1,S2) to open
    :param histname: histogram name
    :param named_parameters: list of config settings to pass to .format on histname and filename
    """

    @staticmethod
    def _get_json_spectrum(fn):
        """
        Translates bbf-style JSON files to spectra.
        units are keV and /kev*day*kg
        """
        contents = json.load(open(fn, "r"))
        logging.debug(contents["description"])
        esyst = contents["coordinate_system"][0][1]
        ret = interp1d(
            np.linspace(*esyst), contents["map"],
            bounds_error=False, fill_value=0.)
        return ret

    def build_histogram(self):
        logging.debug("building a hist")
        format_dict = {
            k: self.config[k]
            for k in self.config.get('named_parameters', [])
        }
        templatename = self.config['templatename'].format(**format_dict)
        histname = self.config['histname'].format(**format_dict)

        spectrum = self.config["spectrum"]
        if type(spectrum) is str:
            spectrum = self._get_json_spectrum(spectrum.format(**format_dict))

        slice_args = self.config.get("slice_args", {})
        if type(slice_args) is dict:
            slice_args = [slice_args]

        h = template_to_multihist(templatename, histname)

        # Perform E-scaling
        ebins = h.bin_edges[0]
        ecenters = 0.5 * (ebins[1::] + ebins[0:-1])
        ediffs = (ebins[1::] - ebins[0:-1])
        h.histogram = h.histogram * (spectrum(ecenters) * ediffs)[:, None, None]
        h = h.sum(axis=0)  # Remove energy-axis
        logging.debug("source", "mu ", h.n)

        for sa in slice_args:
            slice_axis = sa.get("slice_axis", None)
            # Decide if you wish to sum the histogram into lower dimensions or
            sum_axis = sa.get("sum_axis", False)

            slice_axis_limits = sa.get("slice_axis_limits", [0, 0])
            collapse_axis = sa.get('collapse_axis', None)
            collapse_slices = sa.get('collapse_slices', None)

            if (slice_axis is not None):
                if sum_axis:
                    h = h.slicesum(
                        axis=slice_axis,
                        start=slice_axis_limits[0],
                        stop=slice_axis_limits[1])
                else:
                    h = h.slice(
                        axis=slice_axis,
                        start=slice_axis_limits[0],
                        stop=slice_axis_limits[1])

            if collapse_axis is not None:
                if collapse_slices is None:
                    raise ValueError(
                        "To collapse you must supply collapse_slices")
                h = h.collapse_bins(collapse_slices, axis=collapse_axis)

        self.dtype = []
        for n, _ in self.config['analysis_space']:
            self.dtype.append((n, float))
        self.dtype.append(('source', int))

        # Fix the bin sizes
        if can_check_binning:
            # Deal with people who have log10'd their bins
            for axis_i in self.config.get('log10_bins', []):
                h.bin_edges[axis_i] = 10 ** h.bin_edges[axis_i]

            # Check if the histogram bin edges are correct
            for axis_i, (_, expected_bin_edges) in enumerate(
                    self.config['analysis_space']):
                expected_bin_edges = np.array(expected_bin_edges)
                seen_bin_edges = h.bin_edges[axis_i]
                # If 1D, hist1d returns bin_edges straight, not as list
                if len(self.config['analysis_space']) == 1:
                    seen_bin_edges = h.bin_edges
                if len(seen_bin_edges) != len(expected_bin_edges):
                    raise ValueError(
                        "Axis %d of histogram %s in root file %s has %d bin edges, but expected %d"
                        % (
                            axis_i, histname, templatename, len(seen_bin_edges),
                            len(expected_bin_edges)))
                try:
                    np.testing.assert_almost_equal(
                        seen_bin_edges / expected_bin_edges,
                        np.ones_like(seen_bin_edges),
                        decimal=4)
                except AssertionError:
                    warnings.warn(
                        "Axis %d of histogram %s in root file %s has bin edges %s, "
                        "but expected %s. Since length matches, setting it expected values..."
                        % (
                            axis_i, histname, templatename, seen_bin_edges,
                            expected_bin_edges))
                    h.bin_edges[axis_i] = expected_bin_edges

        self._bin_volumes = h.bin_volumes()  # TODO: make alias
        # Shouldn't be in HistogramSource... anyway
        self._n_events_histogram = h.similar_blank_histogram()

        if self.config.get('normalise_template', False):
            h /= h.n

        h *= self.config.get('histogram_multiplier', 1)

        # Convert h to density...
        if self.config.get('in_events_per_bin'):
            h.histogram /= h.bin_volumes()
        self.events_per_day = (h.histogram * self._bin_volumes).sum()
        logging.debug("source evt per day is", self.events_per_day)

        # ... and finally to probability density
        h.histogram /= self.events_per_day
        self._pdf_histogram = h

    def simulate(self, n_events):
        dtype = []
        for n, _ in self.config['analysis_space']:
            dtype.append((n, float))
        dtype.append(('source', int))
        ret = np.zeros(n_events, dtype=dtype)
        # t = self._pdf_histogram.get_random(n_events)
        h = self._pdf_histogram * self._bin_volumes
        t = h.get_random(n_events)
        for i, (n, _) in enumerate(self.config.get('analysis_space')):
            ret[n] = t[:, i]
        return ret
