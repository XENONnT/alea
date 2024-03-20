from pydantic import BaseModel, validator, validate_call, ConfigDict
from typing import List, Dict, Optional, Union, Any, Literal, Callable, Iterable
import numpy as np
from scipy import stats
from copy import deepcopy
from scipy import interpolate, special
import pandas as pd
from tqdm import tqdm
import numba
from multihist import Hist1d


def energy_res(energy, a=25.8, b=1.429):
    """
    Return energy resolution in keV.

    :param energy: true energy in keV
    :return: energy resolution in keV
    """
    # Reference for the values of a,b:
    # xenon:xenonnt:analysis:ntsciencerun0:g1g2_update#standard_gaussian_vs_skew-gaussian_yue
    return (np.sqrt(energy) * a + energy * b) / 100


@validate_call
def smearing_hist_gaussian(
    hist: Any,
    smearing_a: float,
    smearing_b: float,
    bins: Optional[Iterable[float]] = None,
):
    """
    Smear a histogram. This allows for non-uniform histogram binning.

    :param hist: the spectrum we want to smear
    :param bins: bin edges of the returned spectrum
    :return: smeared histogram in the same unit as input spectrum
    """
    assert isinstance(hist, Hist1d), "Only Hist1d object is supported"
    if bins is None:
        # set the bins to the bin edges of the input histogram
        bins = hist.bin_edges
    bins = np.array(bins)

    e_true_s, rates, bin_volumes = hist.bin_centers, hist.histogram, hist.bin_volumes()
    mask = np.where(e_true_s > 0)
    e_true_s = e_true_s[mask]
    rates = rates[mask]
    bin_volumes = bin_volumes[mask]

    e_smeared_s = 0.5 * (bins[1:] + bins[:-1])
    smeared = np.zeros_like(e_smeared_s)

    for idx, e_smeared in enumerate(e_smeared_s):
        probs = (
            stats.norm.pdf(
                e_smeared,
                loc=e_true_s,
                scale=energy_res(e_true_s, smearing_a, smearing_b),
            )
            * bin_volumes
        )
        smeared[idx] = np.sum(probs * rates)

    hist_smeared = Hist1d.from_histogram(smeared, bins)

    return hist_smeared


ERF_X = np.linspace(-5, 5, 1000)
ERF_X = np.concatenate([[-1e6], ERF_X, [1e6]])
ERF_Y = special.erf(ERF_X)


@numba.njit()
def interp_erf(x):
    return np.interp(x, ERF_X, ERF_Y)


@numba.njit()
def skewgauss(x, a, w, loc, N):
    t1 = 1 / (w * (2 * np.pi) ** 0.5)
    t2 = np.exp(-((x - loc) ** 2) / (2 * w**2))
    t3 = 1 + interp_erf(a * (x - loc) / w / 2**0.5)
    return N * t1 * t2 * t3


@numba.njit()
def skew_over_E(E, sa=1.999, sp=-1.256):
    """
    Return skewness for skew gaussian.
    Ref: xenon:xenonnt:analysis:ntsciencerun0:g1g2_update#standard_gaussian_vs_skew-gaussian_yue

    :param E: peak true energy in keV
    :return: skewness, dimensionless.
    :wc: amplitude of the skewness function
    :wd: power of the skewness function
    """
    skewness = sa * (E**sp) * E
    return skewness


@numba.njit()
def w_over_E(E, wa=37.2e-2, wb=4.36e-3):
    """
    Return width for skew gaussian. When skewness is 0, width is then energy resolution for regular gaussian.
    Ref: xenon:xenonnt:analysis:ntsciencerun0:g1g2_update#standard_gaussian_vs_skew-gaussian_yue

    :param E: peak energy in keV
    :return: width , dimensionless
    """
    width = (wa * E ** (-0.5) + wb) * E
    return width


@numba.njit()
def _skew_gaussian_loc(e, a, w):
    """
    Get skewed Gaussian location from mean value and skewness

    :param e: mean energy in keV
    :param a: skewness, dimensionless
    :param w: skew Gaussian width in keV
    :return: location of skewed Gaussian in keV
    """
    loc = e - np.sqrt(2 / np.pi) * a * w / np.sqrt(1 + a**2)
    return loc


@numba.njit()
def skew_gaussian_loc(e, sa, sp, wa, wb):
    """
    Get skewed Gaussian location from the mean energy

    :param e: mean energy in keV
    :return: location of skewed Gaussian in keV
    """
    # width in kev
    w = w_over_E(e, wa=wa, wb=wb)
    # dimensionless skewness
    a = skew_over_E(e, sa=sa, sp=sp)

    loc = _skew_gaussian_loc(e, a, w)
    return loc


@numba.njit()
def _smear_skew_array(e_true_s, rates, bin_volumes, bins, sa, sp, wa, wb):
    mask = np.where(e_true_s > 0)
    e_true_s = e_true_s[mask]
    rates = rates[mask]
    bin_volumes = bin_volumes[mask]

    # convert to skew gaussian params
    skew_loc_s = skew_gaussian_loc(e_true_s, sa, sp, wa, wb)
    skew_scale_s = w_over_E(e_true_s, wa, wb)
    skewness_s = skew_over_E(e_true_s, sa, sp)

    e_smeared_s = 0.5 * (bins[1:] + bins[:-1])
    smeared = np.zeros_like(e_smeared_s)

    for idx, e_smeared in enumerate(e_smeared_s):
        # probs = stats.skewnorm.pdf(e_smeared, a=skewness_s, loc=skew_loc_s, scale=skew_scale_s) * bin_volumes
        probs = (
            skewgauss(e_smeared, a=skewness_s, loc=skew_loc_s, w=skew_scale_s, N=1)
            * bin_volumes
        )
        smeared[idx] = np.sum(probs * rates)

    return smeared


@validate_call
def smearing_hist_skew_gaussian(
    hist: Any, sa: float, sp: float, wa: float, wb: float, bins=None
):
    """
    Smear a histogram using skew gaussian. This allows for non-uniform histogram binning.

    :param hist: the spectrum we want to smear
    :param bins: bin edges of the returned spectrum
    :param add_bias: boolean, include reconstruction bias in spectrum modeling if True. Default True.
    :return: smeared histogram using skew gaussian in the same unit as input spectrum
    """
    assert isinstance(hist, Hist1d), "Only Hist1d object is supported"
    if bins is None:
        # set the bins to the bin edges of the input histogram
        bins = hist.bin_edges

    e_true_s, rates, bin_volumes = hist.bin_centers, hist.histogram, hist.bin_volumes()
    smeared = _smear_skew_array(e_true_s, rates, bin_volumes, bins, sa, sp, wa, wb)

    hist_smeared = Hist1d.from_histogram(smeared, bins)
    return hist_smeared


@validate_call
def biasing_hist_arctan(hist: Any, A: float = 0.01977, k: float = 0.01707):
    """
    Apply a constant bias to a histogram

    :param hist: the spectrum we want to apply the bias to
    :param bias: the bias to apply to the spectrum
    :return: the spectrum with the bias applied
    """
    assert isinstance(hist, Hist1d), "Only Hist1d object is supported"
    true_energy = hist.bin_centers
    h_bias = deepcopy(hist)
    bias_derivative = A * k / (1 + k**2 * true_energy**2)
    h_bias.histogram *= 1 / (1 + bias_derivative)
    return h_bias


@validate_call
def biasing_hist_sigmoid(hist: Any, A: float =0.0513, k: float =0.0247):
    assert isinstance(hist, Hist1d), "Only Hist1d object is supported"
    true_energy = hist.bin_centers
    h_bias = deepcopy(hist)
    bias_derivative = (
        A * k * np.exp(-k * true_energy) / (1 + np.exp(-k * true_energy)) ** 2
    )
    h_bias.histogram *= 1 / (1 + bias_derivative)
    return h_bias


@validate_call
def efficiency_hist_constant(hist: Any, efficiency: float):
    """
    Apply a constant efficiency to a histogram

    :param hist: the spectrum we want to apply the efficiency to
    :param efficiency: the efficiency to apply to the spectrum
    :return: the spectrum with the efficiency applied
    """
    assert isinstance(hist, Hist1d), "Only Hist1d object is supported"
    assert 0 <= efficiency <= 1, "Efficiency must be between 0 and 1"
    return hist * efficiency


MODELS: Dict[str, Dict[str, Callable]] = {
    "smearing": {
        "gaussian": smearing_hist_gaussian,
        "skew_gaussian": smearing_hist_skew_gaussian,
    },
    "bias": {"arctan": biasing_hist_arctan, "sigmoid": biasing_hist_sigmoid},
    "efficiency": {"constant": efficiency_hist_constant},
}


# input: model name, parameters, transformation mode
class Transformation(BaseModel):
    parameters: Dict[str, float]
    action: Literal["smearing", "bias", "efficiency"]
    model: str

    @validator("model")
    @classmethod
    def check_model(cls, v, values):
        if v not in MODELS[values["action"]]:
            raise ValueError(f"Model {v} not found for action {values['action']}")
        return v

    def apply_transformation(self, histogram: Hist1d):
        chosen_model = MODELS[self.action][self.model]
        return chosen_model(histogram, **self.parameters)
