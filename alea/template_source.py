import warnings
import logging

import blueice
import numpy as np
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
