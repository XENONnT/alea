from pydantic import BaseModel, validator
from typing import Dict, Optional, Any, Literal, Callable
import numpy as np
from scipy import stats
from multihist import Hist1d
from scipy.interpolate import interp1d


def energy_res(energy, a: float, b: float):
    """Return energy resolution in keV.

    Args:
        energy: True energy in keV.
        a: First resolution parameter.
        b: Second resolution parameter.

    Returns:
        Energy resolution in keV.

    """
    # Reference for the values of a,b:
    # xenon:xenonnt:analysis:ntsciencerun0:g1g2_update#standard_gaussian_vs_skew-gaussian_yue
    return (np.sqrt(energy) * a + energy * b) / 100


def smearing_mono_gaussian(
    hist: Any,
    smearing_a: float,
    smearing_b: float,
    peak_energy: float,
    bins: Optional[np.ndarray] = None,
):
    """Smear a mono-energetic peak with a Gaussian using CDF difference method.

    Args:
        hist: The histogram to smear.
        smearing_a: First smearing parameter.
        smearing_b: Second smearing parameter.
        peak_energy: Energy of the mono-energetic peak.
        bins: Optional bin edges for the returned histogram. Defaults to None.

    Returns:
        Smeared histogram.

    Raises:
        ValueError: If bins has less than 2 elements.

    """

    if bins is None:
        bins = hist.bin_edges
        bin_centers = hist.bin_centers
    else:
        # use the bins that set by the user
        bins = np.array(bins)
        if bins.size <= 1:
            raise ValueError("bins must have at least 2 elements")
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Calculate bin widths
    bin_widths = np.diff(bins)

    # Create array to hold the result
    data = np.zeros_like(bin_centers)

    # Calculate the resolution at the peak energy
    scale = energy_res(peak_energy, smearing_a, smearing_b)

    # For each bin, calculate CDF difference
    for i, center in enumerate(bin_centers):
        half_width = bin_widths[i] * 0.5
        upper_edge = center + half_width
        lower_edge = center - half_width

        # CDF difference gives probability mass in this bin
        data[i] = stats.norm.cdf(upper_edge, loc=peak_energy, scale=scale) - stats.norm.cdf(
            lower_edge, loc=peak_energy, scale=scale
        )

    # Create and populate the histogram
    hist_smeared = Hist1d(data=np.zeros_like(data), bins=bins)
    hist_smeared.histogram = data

    return hist_smeared


def smearing_hist_gaussian(
    hist: Any,
    smearing_a: float,
    smearing_b: float,
    bins: Optional[np.ndarray] = None,
):
    """Smear a histogram with Gaussian using CDF difference method.

    Args:
        hist: The spectrum we want to smear.
        smearing_a: First smearing parameter.
        smearing_b: Second smearing parameter.
        bins: Bin edges of the returned spectrum. Defaults to None.

    Returns:
        Smeared histogram in the same unit as input spectrum.

    Raises:
        AssertionError: If hist is not a Hist1d object.
        ValueError: If bins has less than 2 elements.

    """
    assert isinstance(hist, Hist1d), "Only Hist1d object is supported"
    if bins is None:
        # set the bins to the bin edges of the input histogram
        bins = hist.bin_edges
    elif bins.size <= 1:
        raise ValueError("bins must have at least 2 elements")
    bins = np.array(bins)

    e_true_s, rates, bin_volumes = hist.bin_centers, hist.histogram, hist.bin_volumes()
    mask = np.where(e_true_s > 0)
    e_true_s = e_true_s[mask]
    rates = rates[mask]
    bin_volumes = bin_volumes[mask]

    e_smeared_s = 0.5 * (bins[1:] + bins[:-1])  # Centers of output bins
    bin_widths = bins[1:] - bins[:-1]  # Widths of output bins
    smeared = np.zeros_like(e_smeared_s)

    for idx, e_smeared in enumerate(e_smeared_s):
        # Calculate bin edges for CDF evaluation
        half_width = bin_widths[idx] * 0.5
        upper_edge = e_smeared + half_width
        lower_edge = e_smeared - half_width

        # Use CDF difference instead of PDF
        scales = energy_res(e_true_s, smearing_a, smearing_b)

        # CDF at upper edge minus CDF at lower edge for each input energy
        probs = (
            stats.norm.cdf(upper_edge, loc=e_true_s, scale=scales)
            - stats.norm.cdf(lower_edge, loc=e_true_s, scale=scales)
        ) * bin_volumes

        smeared[idx] = np.sum(probs * rates)

    hist_smeared = Hist1d.from_histogram(smeared, bins)

    return hist_smeared


def biasing_hist_arctan(hist, A: float, k: float, B: float):
    """Apply bias and Jacobian correction to a 1D histogram, and interpolate the corrected spectrum
    onto measured energy E'.

    - Keeps the original bin edges
    - Returns a new Hist1d object
    - bin_centers represent measured energy (E')
    - histogram is interpolated from corrected f(E)

    Parameters
    ----------
    hist : Hist1d
        Original histogram in true energy domain E
    A, k, B : float
        Bias function parameters:
            (E' - E) / E = A * arctan(kE) + B

    Returns
    -------
    Hist1d
        New histogram with interpolated and corrected values, x-axis is E'

    """
    assert isinstance(hist, Hist1d), "Only Hist1d object is supported"

    E = hist.bin_centers
    f_E = hist.histogram.astype(float)

    # Bias function and Jacobian
    b_E = A * np.arctan(k * E) + B
    b_prime_E = A * k / (1 + (k * E) ** 2)
    jacobian = 1 + b_E + E * b_prime_E

    # E' = E * (1 + b(E))
    E_prime = E * (1 + b_E)

    # Corrected spectrum: g(E') = f(E) / dE'/dE
    f_corrected = f_E / jacobian

    # Interpolate onto original bin_centers (which we now reinterpret as measured energy)
    interp = interp1d(E_prime, f_corrected, bounds_error=False, fill_value=0, kind="linear")

    E_meas_bins = hist.bin_centers
    f_interp = interp(E_meas_bins)

    new_hist = Hist1d.from_histogram(histogram=f_interp, bin_edges=hist.bin_edges)
    return new_hist


def efficiency_hist_constant(hist: Any, efficiency_constant: float):
    """Apply a constant efficiency to a histogram.

    Args:
        hist: The spectrum we want to apply the efficiency to.
        efficiency_constant: The efficiency to apply to the spectrum.

    Returns:
        The spectrum with the efficiency applied.

    Raises:
        AssertionError: If hist is not a Hist1d object
        or if efficiency_constant is not between 0 and 1.

    """
    assert isinstance(hist, Hist1d), "Only Hist1d object is supported"
    assert 0 <= efficiency_constant <= 1, "Efficiency must be between 0 and 1"
    hist.histogram = hist.histogram * efficiency_constant
    return hist


MODELS: Dict[str, Dict[str, Callable]] = {
    "smearing": {
        "gaussian": smearing_hist_gaussian,
        "mono_gaussian": smearing_mono_gaussian,
    },
    "bias": {"arctan": biasing_hist_arctan},
    "efficiency": {
        "constant": efficiency_hist_constant,
    },
}


# input: model name, parameters, transformation mode
class Transformation(BaseModel):
    parameters: Dict[str, float]
    action: Literal["bias", "smearing", "efficiency"]
    model: str

    @validator("model")
    @classmethod
    def check_model(cls, v, values):
        """Check if the model exists for the given action.

        Args:
            v: The model name.
            values: The values dictionary containing the action.

        Returns:
            The model name if it exists.

        Raises:
            ValueError: If the model does not exist for the given action.

        """
        if v not in MODELS[values["action"]]:
            raise ValueError(f"Model {v} not found for action {values['action']}")
        return v

    def apply_transformation(self, histogram: Hist1d):
        """Apply the transformation to a histogram.

        Args:
            histogram: The histogram to transform.

        Returns:
            The transformed histogram.

        """
        chosen_model = MODELS[self.action][self.model]
        return chosen_model(histogram, **self.parameters)
