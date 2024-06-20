import logging
from typing import List, Dict, Optional, Union

import numpy as np
from scipy.interpolate import interp1d
from blueice import HistogramPdfSource
from multihist import Hist1d
from inference_interface import template_to_multihist

from alea.utils import load_json

logging.basicConfig(level=logging.INFO)
can_check_binning = True


class TemplateSource(HistogramPdfSource):
    """A source defined with a template histogram. The parameters are set in self.config.
    "templatename", "histname", "analysis_space" must be in self.config.

    Attributes:
        config (dict): The configuration of the source.
        dtype (list): The data type of the source.
        _bin_volumes (numpy.ndarray): The bin volumes of the source.
        _n_events_histogram (multihist.MultiHistBase):
            The histogram of the number of events of the source.
        events_per_day (float): The number of events per day of the source.
        _pdf_histogram (multihist.MultiHistBase):
            The histogram of the probability density function of the source.

    Args:
        config (dict): The configuration of the source.
        templatename: Hdf5 file to open.
        histname: Histogram name.
        named_parameters (list):
            List of config setting names to pass to .format on histname and filename.
        normalise_template (bool): Normalise the template histogram.
        in_events_per_bin (bool):
            If True, histogram is in events per day / bin.
            If False or absent, histogram is already pdf.
        histogram_scale_factor (float): Multiply histogram by this number
        convert_to_uniform (bool): Convert the histogram to a uniform per bin distribution.
        log10_bins (list):
            List of axis numbers.
            If True, bin edges on this axis in the hdf5 file are log10() of the actual bin edges.

    """

    def __init__(self, config: Dict, *args, **kwargs):
        """Initialize the TemplateSource."""
        # override the default interpolation method
        if "pdf_interpolation_method" not in config:
            config["pdf_interpolation_method"] = "piecewise"
        super().__init__(config, *args, **kwargs)

    def _check_binning(self, h, histogram_info: str):
        """Check if the histogram"s bin edges are the same to analysis_space.

        Args:
            h (multihist.MultiHistBase): The histogram to check.
            histogram_info (str): Information of the histogram.

        """
        # Deal with people who have log10"d their bins
        for axis_i in self.config.get("log10_bins", []):
            h.bin_edges[axis_i] = 10 ** h.bin_edges[axis_i]

        analysis_space = self.config["analysis_space"]
        if len(analysis_space) != h.histogram.ndim:
            raise ValueError(
                f"Analysis space {analysis_space} has {len(analysis_space)} axes, "
                f"but histogram {histogram_info} has {h.histogram.ndim} axes."
            )

        # Check if the histogram bin edges are correct
        for axis_i, (_, expected_bin_edges) in enumerate(analysis_space):
            expected_bin_edges = np.array(expected_bin_edges)
            # If 1D, hist1d returns bin_edges straight, not as list
            if isinstance(h, Hist1d):
                seen_bin_edges = h.bin_edges
            else:
                seen_bin_edges = h.bin_edges[axis_i]
            logging.debug("axis_i: " + str(axis_i))
            logging.debug("expected_bin_edges: " + str(expected_bin_edges))
            logging.debug("seen_bin_edges: " + str(seen_bin_edges))
            logging.debug("h.bin_edges type" + str(h.bin_edges))
            err_msg = (
                f"Axis {axis_i:d} of histogram {histogram_info} "
                f"has bin edges {seen_bin_edges}, but expected {expected_bin_edges}."
            )
            if len(seen_bin_edges) != len(expected_bin_edges):
                raise ValueError(err_msg)
            np.testing.assert_almost_equal(
                seen_bin_edges, expected_bin_edges, decimal=2, err_msg=err_msg
            )

    @property
    def format_named_parameters(self):
        """Get the named parameters in the config to dictionary format."""
        format_named_parameters = {
            k: self.config[k] for k in self.config.get("named_parameters", [])
        }
        return format_named_parameters

    def build_histogram(self):
        """Build the histogram of the source."""
        templatename = self.config["templatename"].format(**self.format_named_parameters)
        histname = self.config["histname"].format(**self.format_named_parameters)
        h = template_to_multihist(templatename, histname)
        if np.min(h.histogram) < 0:
            raise AssertionError(f"There are bins for source {templatename} with negative entries.")

        if self.config.get("normalise_template", False):
            h /= h.n
        if not self.config.get("in_events_per_bin", True):
            h.histogram *= h.bin_volumes()

        h = self.apply_slice_args(h)

        # Fix the bin sizes
        if can_check_binning:
            histogram_info = f"{histname:s} in hdf5 file {templatename:s}"
            self._check_binning(h, histogram_info)

        self.set_dtype()
        self.set_pdf_histogram(h)

    def apply_slice_args(self, h, slice_args: Optional[Union[List[Dict], Dict]] = None):
        """Apply slice arguments to the histogram.

        Args:
            h (multihist.MultiHistBase): The histogram to apply the slice arguments to.
            slice_args (dict): The slice arguments to apply.
                The sum_axis, slice_axis, and slice_axis_limits are supported.

        """

        # if slice_args is not specified, use the one in the config
        if slice_args is None:
            slice_args = self.config.get("slice_args", [{}])
        # if slice_args is a dict, convert it to a list
        if isinstance(slice_args, dict):
            slice_args = [slice_args]
        # if slice_args is empty, return
        if (slice_args is None) or (len(slice_args) == 0) or (slice_args == [{}]):
            return h
        # check if slice_args is a list of dicts, and if each dict has slice_axis
        if not isinstance(slice_args, list):
            raise ValueError("slice_args must be a list or a dict")
        if any(not isinstance(sa, dict) for sa in slice_args):
            raise ValueError("slice_args must be a list of dicts")
        if any("slice_axis" not in sa for sa in slice_args):
            raise ValueError("slice_axis must be specified in slice_args")

        for sa in slice_args:
            # read slice_axis, sum_axis, and slice_axis_limits from slice_args
            slice_axis = sa["slice_axis"]
            sum_axis = sa.get("sum_axis", False)
            bin_edges = h.bin_edges[h.get_axis_number(slice_axis)]
            slice_axis_limits = sa.get("slice_axis_limits", [bin_edges[0], bin_edges[-1]])

            # slice and/or sum over axis
            if sum_axis:
                logging.debug(
                    f"Slice and sum over axis {slice_axis} from {slice_axis_limits[0]} "
                    f"to {slice_axis_limits[1]}"
                )
                h = h.slicesum(
                    start=slice_axis_limits[0], stop=slice_axis_limits[1], axis=slice_axis
                )
            else:
                logging.debug(
                    f"Slice over axis {slice_axis} from {slice_axis_limits[0]} "
                    f"to {slice_axis_limits[1]}"
                )
                logging.debug(f"Normalization before slicing: {h.n}.")
                h = h.slice(start=slice_axis_limits[0], stop=slice_axis_limits[1], axis=slice_axis)
                logging.debug(f"Normalization after slicing: {h.n}.")

        if isinstance(h, Hist1d):
            if self.config["pdf_interpolation_method"] != "piecewise":
                raise ValueError(
                    "pdf_interpolation_method must be piecewise for 1D histograms. "
                    "This is required by blueice."
                )
        return h

    def set_dtype(self):
        """Set the data type of the source."""
        self.dtype = []
        for n, _ in self.config["analysis_space"]:
            self.dtype.append((n, float))
        self.dtype.append(("source", int))

    def set_pdf_histogram(self, h):
        """Set the histogram of the probability density function of the source."""
        self._bin_volumes = h.bin_volumes()
        # Should not be in HistogramSource... anyway
        self._n_events_histogram = h.similar_blank_histogram()

        histogram_scale_factor = self.config.get("histogram_scale_factor", 1)
        h *= histogram_scale_factor
        logging.debug(
            f"Multiplying histogram with histogram_scale_factor {histogram_scale_factor}. "
            f"Histogram is now normalised to {h.n}."
        )

        self.events_per_day = h.n
        logging.debug(f"events_per_day: {self.events_per_day}")

        # Convert h to density...
        if self.config.get("in_events_per_bin", True):
            h.histogram /= self._bin_volumes

        # ... and finally to probability density
        if self.events_per_day > 0:
            h.histogram /= self.events_per_day
        else:
            logging.warn("Sum of histogram is zero.")

        if self.config.get("convert_to_uniform", False):
            h.histogram = 1.0 / self._bin_volumes

        self._pdf_histogram = h
        logging.debug(f"Setting _pdf_histogram normalised to {h.n}.")

    def simulate(self, n_events: int):
        """Simulate events from the source.

        Args:
            n_events (int): The number of events to simulate.

        Returns:
            numpy.ndarray: The simulated events.

        """
        ret = np.zeros(n_events, dtype=self.dtype)
        h = self._pdf_histogram * self._bin_volumes
        t = h.get_random(n_events)
        for i, (n, _) in enumerate(self.config["analysis_space"]):
            ret[n] = t[:, i]
        return ret


class CombinedSource(TemplateSource):
    """Source that is a weighted sums of histograms. Useful e.g. for safeguard. The first histogram
    is the base histogram, and the rest are added to it with weights. The weights can be set as
    shape parameters in the config.

    Args:
        weights: Weights of the 2nd to the last histograms.
        histnames: List of filenames containing the histograms.
        templatenames: List of names of histograms within the hdf5 files.

    """

    def build_histogram(self):
        """Build the histogram of the source."""
        if not self.config.get("in_events_per_bin", True):
            raise ValueError("CombinedSource does not support in_events_per_bin=False")
        # Check if all the necessary parameters, like weights are specified
        if "weight_names" not in self.config:
            raise ValueError("weight_names must be specified")
        find_weight = [weight_name in self.config for weight_name in self.config["weight_names"]]
        if not all(find_weight):
            missing_weight = self.config["weight_names"][find_weight]
            raise ValueError(f"All weight_names must be specified, but {missing_weight} are not. ")
        weights = [self.config[weight_name] for weight_name in self.config["weight_names"]]
        histnames = self.config["histnames"]
        templatenames = self.config["templatenames"]
        if len(histnames) != len(templatenames):
            raise ValueError("histnames and templatenames must be the same length")
        if len(weights) + 1 != len(templatenames):
            raise ValueError("weights must be 1 shorter than histnames, templatenames")

        slice_args = self.config.get("slice_args", {})
        if isinstance(slice_args, dict):
            slice_args = [slice_args]
        if all(isinstance(sa, dict) for sa in slice_args):
            # Recognize as a single slice_args for all histograms
            slice_argss = [slice_args] * len(histnames)
        else:
            # Recognize as a list of slice_args for each histogram
            slice_argss = slice_args

        histograms = []
        slice_fractions = []
        for histname, templatename, slice_args in zip(histnames, templatenames, slice_argss):
            templatename = templatename.format(**self.format_named_parameters)
            histname = histname.format(**self.format_named_parameters)
            h = template_to_multihist(templatename, histname)
            slice_fraction = 1.0 / h.n
            if h.n == 0.0:
                slice_fraction = 0.0
            h = self.apply_slice_args(h, slice_args)
            slice_fraction *= h.n
            slice_fractions.append(slice_fraction)

            # Avoid dividing by zero for empty PDFs (if mu=0, does not matter if unnormalised)
            if h.n <= 0.0:
                h *= 0.0
            else:
                h /= h.n
            histograms.append(h)
        logging.debug("slice fractions are", slice_fractions)

        for i in range(len(weights)):
            h_comp = histograms[0].similar_blank_histogram()
            h = histograms[i + 1]
            hslices = []
            for j, bincs in enumerate(h.bin_centers()):
                hsliced = h_comp.get_axis_bin_index(bincs[0], j)
                hsliceu = h_comp.get_axis_bin_index(bincs[-1], j) + 1
                hslices.append(slice(hsliced, hsliceu, 1))
            h_comp[tuple(hslices)] += h.histogram
            histograms[0] += h_comp * weights[i]
        h = histograms[0]

        logging.debug("Setting zero for NaNs and Infs in combined template.")
        h.histogram[np.isinf(h.histogram)] = 0.0
        h.histogram[np.isnan(h.histogram)] = 0.0
        h.histogram[h.histogram < 0] = 0.0

        # Set pdf values that are below 0 to zero:
        if np.min(h.histogram) < 0:
            raise AssertionError(f"There are bins for source {templatename} with negative entries.")
        # Fix the bin sizes
        if can_check_binning:
            histogram_info = f"combined template {histnames} in hdf5 file {templatenames}"
            self._check_binning(h, histogram_info)

        self.set_dtype()
        self.set_pdf_histogram(h)


class SpectrumTemplateSource(TemplateSource):
    """Reweighted template source by 1D spectrum. The first axis of the template is assumed to be
    reweighted.

    Args:
        spectrum_name:
            Name of bbf json-like spectrum file

    """

    @staticmethod
    def _get_json_spectrum(filename: str):
        """Translates bbf-style JSON files to spectra.

        Args:
            filename (str): Name of the JSON file.

        Todo:
            Define the format of the JSON file clearly.

        """
        contents = load_json(filename)
        if "description" in contents:
            logging.debug(contents["description"])
        if "coordinate_system" not in contents:
            raise ValueError("Coordinate system not in JSON file.")
        if "map" not in contents:
            raise ValueError("Map not in JSON file.")
        ret = interp1d(
            contents["coordinate_system"], contents["map"], bounds_error=False, fill_value=0.0
        )
        return ret

    def build_histogram(self):
        """Build the histogram of the source."""
        templatename = self.config["templatename"].format(**self.format_named_parameters)
        histname = self.config["histname"].format(**self.format_named_parameters)
        h = template_to_multihist(templatename, histname)

        if "spectrum_name" not in self.config:
            raise ValueError("spectrum_name not in config")
        spectrum = self._get_json_spectrum(
            self.config["spectrum_name"].format(**self.format_named_parameters)
        )

        # Perform scaling, the first axis is assumed to be reweighted
        # The spectrum is assumed to be probability density (in per the unit of first axis).
        axis = 0
        # h = h.normalize(axis=axis)
        bin_edges = h.bin_edges[axis]
        bin_centers = h.bin_centers(axis=axis)
        slices = [None] * h.histogram.ndim
        slices[axis] = slice(None)
        h.histogram = h.histogram * (spectrum(bin_centers) * np.diff(bin_edges))[tuple(slices)]

        if np.min(h.histogram) < 0:
            raise AssertionError(f"There are bins for source {templatename} with negative entries.")

        if self.config.get("normalise_template", False):
            h /= h.n
        if not self.config.get("in_events_per_bin", True):
            h.histogram *= h.bin_volumes()

        h = self.apply_slice_args(h)

        # Fix the bin sizes
        if can_check_binning:
            histogram_info = f"reweighted {histname:s} in hdf5 file {templatename:s}"
            self._check_binning(h, histogram_info)

        self.set_dtype()
        self.set_pdf_histogram(h)
