import numpy as np
from typing import List, Dict, Optional, Union

from inference_interface import template_to_multihist
from blueice import HistogramPdfSource
from multihist import Hist1d
from alea.ces_functions import apply_transformation

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
    def _check_histogram(self, h):
        """
        Check if the histogram has expected binning
        """
        # We only take 1d histogram in the ces axes
        if isinstance(h, Hist1d):
            raise ValueError("Only Hist1d object is supported")
        if len(self.analysis_space) != 1:
            raise ValueError("Only 1d analysis space is supported")
        if list(self.analysis_space[0].keys())[0] != "ces":
            raise ValueError("The analysis space must be ces")
        if np.min(h.histogram) < 0:
            raise AssertionError(f"There are bins for source {self.templatename} with negative entries.")
        
        # check if the histogram contains the analysis space. The range of histogram should be larger
        # than the analysis space. The min and max of the histogram should be smaller/larger than the
        # min/max of the analysis space
        histogram_max = np.max(h.bin_edges)
        histogram_min = np.min(h.bin_edges)
        if self.min_e < histogram_min or self.max_e > histogram_max:
            raise ValueError(f"The histogram edge ({histogram_min},{histogram_max}) \
                does not contain the analysis space ({self.min_e},{self.max_e})")
    def build_hitogram(self):
        """Build the histogram of the source."""
        h = template_to_multihist(self.templatename, self.histname)
        self._check_histogram(h)

        h = apply_transformation(h, self.config, 'smearing', apply_smearing)
        h = apply_transformation(h, self.config, 'bias', apply_bias)
        h = apply_transformation(h, self.config, 'efficiency', apply_efficiency)

            
            
            
        # To avoid confusion, we always load spectrum in unit of events/(t y kev)
        h.histogram /= h.bin_volumes() 
        h /= h.n
        
    
