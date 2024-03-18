from typing import List, Dict, Optional, Union, Literal
import numpy as np
from scipy.interpolate import interp1d

from inference_interface import template_to_multihist
from blueice import HistogramPdfSource
from blueice.exceptions import PDFNotComputedException

from multihist import Hist1d
from alea.ces_functions import Transformation


class CESTemplateSource(HistogramPdfSource):
    def __init__(self, config: Dict, *args, **kwargs):
        """Initialize the TemplateSource."""
        # override the default interpolation method
        if "pdf_interpolation_method" not in config:
            config["pdf_interpolation_method"] = "piecewise"
        super().__init__(config, *args, **kwargs)
        self.analysis_space = self.config["analysis_space"]
        self.templatename = self.config["templatename"]
        self.histname = self.config["histname"]
        self.min_e = np.min(self.analysis_space[0]["ces"])
        self.max_e = np.max(self.analysis_space[0]["ces"])

    def _check_histogram(self, h: Hist1d):
        """
        Check if the histogram has expected binning
        """
        # We only take 1d histogram in the ces axes
        if not isinstance(h, Hist1d):
            raise ValueError("Only Hist1d object is supported")
        if len(self.analysis_space) != 1:
            raise ValueError("Only 1d analysis space is supported")
        if list(self.analysis_space[0].keys())[0] != "ces":
            raise ValueError("The analysis space must be ces")
        if np.min(h.histogram) < 0:
            raise AssertionError(
                f"There are bins for source {self.templatename} with negative entries."
            )

        # check if the histogram contains the analysis space. The range of histogram should be larger
        # than the analysis space. The min and max of the histogram should be smaller/larger than the
        # min/max of the analysis space
        histogram_max = np.max(h.bin_edges)
        histogram_min = np.min(h.bin_edges)
        if self.min_e < histogram_min or self.max_e > histogram_max:
            raise ValueError(
                f"The histogram edge ({histogram_min},{histogram_max}) \
                does not contain the analysis space ({self.min_e},{self.max_e})"
            )

    def create_transformation(
        self, transformation_type: Literal["efficiency", "smearing", "bias"]
    ):
        if self.config.get(f"apply_{transformation_type}", True):
            parameters_key = f"{transformation_type}_parameters"
            model_key = f"{transformation_type}_model"

            if parameters_key not in self.config:
                raise ValueError(
                    f"{transformation_type.capitalize()} parameters are not provided"
                )
            if model_key not in self.config:
                raise ValueError(
                    f"{transformation_type.capitalize()} model is not provided"
                )

            return Transformation(
                parameters=self.config[parameters_key],
                action=transformation_type,
                model=self.config[model_key],
            )
        return None

    def build_hitogram(self):
        """Build the histogram of the source."""
        h = template_to_multihist(self.templatename, self.histname)
        self._check_histogram(h)
        # To avoid confusion, we always normalize the histogram, regardless of the bin volume
        # So the unit is always events/ton/year/keV, the rate multipliers are always in terms of that

        total_integration = np.trapz(h.histogram, h.bin_centers)
        h.histogram /= total_integration

        # Create transformations for efficiency, smearing, and bias
        efficiency_transformation = self.create_transformation("efficiency")
        smearing_transformation = self.create_transformation("smearing")
        bias_transformation = self.create_transformation("bias")

        # Apply the transformations to the histogram
        if efficiency_transformation is not None:
            h = efficiency_transformation.apply(h)
        if smearing_transformation is not None:
            h = smearing_transformation.apply(h)
        if bias_transformation is not None:
            h = bias_transformation.apply(h)

        # Calculate the integration of the histogram after all transformations to estimate the event rate
        # And only from min_e to max_e
        left_edges = h.bin_edges[:-1]
        right_edges = h.bin_edges[1:]
        outside_index = np.where((left_edges < self.min_e) | (right_edges > self.max_e))
        h.histogram[outside_index] = 0

        # Note that it already does what "fraction_in_roi" does in the old code. So no need to calculate that again
        integration_after_transformation_in_roi = np.trapz(h.histogram, h.bin_centers)
        self.events_per_year = (
            integration_after_transformation_in_roi
            * self.config["rate_multiplier"]
            * self.config["fiducial_mass"]
            / 1000
        )
        self.events_per_day = self.events_per_year / 365

        # For pdf, we need to normalize the histogram to 1 again
        h.histogram /= integration_after_transformation_in_roi
        self._pdf_histogram = h

    def simulate(self, n_events: int):
        dtype = [
            ("ces", float),
            ("source", int),
        ]
        ret = np.zeros(n_events, dtype=dtype)
        ret["ces"] = self._pdf_histogram.get_random(n_events)
        return ret

    def pdf(self, *args):
        # override the default interpolation method in blueice (RegularGridInterpolator)
        if not self.pdf_has_been_computed:
            raise PDFNotComputedException(
                "%s: Attempt to call a PDF that has not been computed" % self
            )

        method = self.config["pdf_interpolation_method"]

        if method == "linear":
            if not hasattr(self, "_pdf_interpolator"):
                # First call:
                # Construct a linear interpolator between the histogram bins
                self._pdf_interpolator = interp1d(
                    self._pdf_histogram.bin_centers,
                    self._pdf_histogram.histogram,
                )
            # The interpolator works only within the bin centers region: clip the input data to that.
            # Assuming you've cut the data to the analysis space first (which you should have!)
            # this is equivalent to assuming constant density in the outer half of boundary bins
            bcs = self._pdf_histogram.bin_centers
            clipped_data = np.clip(args, bcs.min(), bcs.max())

            return self._pdf_interpolator(np.transpose(clipped_data))

        elif method == "piecewise":
            return self._pdf_histogram.lookup(*args)

        else:
            raise NotImplementedError(
                "PDF Interpolation method %s not implemented" % method
            )
