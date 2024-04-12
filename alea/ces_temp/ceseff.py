"""
This file provides energy reconstruction and efficiency. Specifically:
  * CES reconstruction
  * Energy resolution for smearing
  * Detection efficiency in CES
  * Cut acceptance in CES

  Authors:
    * Jingqiang Ye <jingqiang.ye@columbia.edu>
    * Yue Ma <yuema@physics.ucsd.edu>
    * Zihao Xu <zx2281@columbia.edu>
    * Evan Shockley <eshockley@physics.ucsd.edu>
"""

import numpy as np
from scipy import stats
from multihist import Hist1d
from copy import deepcopy
from scipy import interpolate, special
import pandas as pd
from tqdm import tqdm
import numba

from nton.utils import data_path

# analysis_range = np.linspace(1, 70, 70)

# Spectrum smearing binning for all backgrounds and signals
# This has to be slightly larger than ROI
# Put a non-integer so that ROI edge can't be the same at least
# If they have the same right bound for instance, then it will just extrapolate to higher energies using the last value
BINS = np.arange(0, 220.5, 0.05)


# ces
def ces(cs1, cs2, kind="nominal"):
    """
    Energy reconstruction.
    :param cs1: in PE
    :param cs2: in PE.
    :param kind: different g1g2 from different analysis. 3 options:

    1. "doke_v8":
    Description: From Henning's Doke plot, no correction, only fit to low energy lines (<250keV)
    Reference: xenon:xenonnt:analysis:ntsciencerun0:g1g2_update#doke_plot_henning

    2. "doke_v8_scale_to_ar":
    Description: From Henning's Doke plot, other CY&LY are corrected so that they have the same biases as Ar37, only fit to low energy lines
    Reference: xenon:xenonnt:analysis:ntsciencerun0:g1g2_update#doke_plot_henning

    3. "nominal":
    Description: Using data from doke_v8_scale_to_ar, but including extra 3.2% systematic errors for Qy, only fit to low energy lines
    Reference: xenon:xenonnt:analysis:ntsciencerun0:g1g2_conservative_error

    The default option is "nominal". This is the nominal setting for LowER
    :return: ces in keV
    """
    # TODO: need to determine the best g1g2
    w = 13.7 / 1000

    if kind == "doke_v8":
        g1 = 0.1532
        g2 = 16.26

    elif kind == "doke_v8_scale_to_ar":
        g1 = 0.1509
        g2 = 16.4885

    elif kind == "nominal":
        g1 = 0.15149
        g2 = 16.45

    else:
        raise ValueError("'kind' can only be 'doke_v8', 'doke_v8_scale_to_ar','nominal'")

    return (cs1 / g1 + cs2 / g2) * w


def energy_res(energy, a=25.8, b=1.429):
    """
    Return energy resolution in keV.

    :param energy: true energy in keV
    :return: energy resolution in keV
    """
    # Reference for the values of a,b:
    # xenon:xenonnt:analysis:ntsciencerun0:g1g2_update#standard_gaussian_vs_skew-gaussian_yue
    return (np.sqrt(energy) * a + energy * b) / 100


# energy bias from data, would be used in the spectrum reshaping
# reference: xenon:xenonnt:yue:lower:smearing_v7#appendix_2reshape_our_energy_spectrum_to_account_for_the_bias
def relative_energy_bias(true_energy, fit_model="arctan"):
    """
    Delta_E/E in absolute value (not percentage)
    Motivated by WFSim reconstruction bias but quantitatively data-driven
    Note that energy bias depends on g1 g2. This bias curve is only for g1 = 0.1509, g2 = 16.4885 (from the scaled-to-ar Doke plot)
    Only valid for [0,300]keV!

    :param true_energy: in keV
    :param fit_model: the fit model to use, 'arctan' or 'sigmoid'. Default 'arctan'.
    :return:
    """

    if fit_model == "arctan":
        A = 0.01977
        k = 0.01707
        return A * np.arctan(k * true_energy)

    elif fit_model == "sigmoid":
        A = 0.0513
        k = 0.0247
        return A * ((1 / (1 + np.exp(-k * true_energy))) - (1 / 2))

    else:
        raise ValueError("only \"arctan\", \"sigmoid\" are allowed for fit_model ")


def biased_energy(true_energy, fit_model="arctan"):
    """
    Get biased energy from true energy.

    :param true_energy: scalar or array, in keV
    :param fit_model: the fit model to use, 'arctan' or 'sigmoid'. Default 'arctan'.
    :return:
    """
    return true_energy * (1 + relative_energy_bias(true_energy, fit_model))


def bias_hist(h, fit_model="arctan"):
    """
    Convert true energy distribution to biased energy distribution due to reconstruction bias. See more details:
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:yue:lower:smearing_v7#appendix_2reshape_our_energy_spectrum_to_account_for_the_bias

    :param h: 1d hist of true energy distribution
    :param fit_model: the fit model to use, 'arctan' or 'sigmoid'. Default 'arctan'.
    :return: 1d hist of biased energy distribution with the same binning
    """
    # can probably take the systematic error between different models, and add a shaping parameter for this?
    true_energy = h.bin_centers
    h_bias = deepcopy(h)

    if fit_model == "arctan":
        A = 0.01977
        k = 0.01707
        bias_derivative = A * k / (1 + k ** 2 * true_energy ** 2)

    elif fit_model == "sigmoid":
        A = 0.0513
        k = 0.0247
        bias_derivative = A * k * np.exp(-k * true_energy) / (1 + np.exp(-k * true_energy)) ** 2

    else:
        raise ValueError("only \"arctan\", \"sigmoid\" are allowed for fit_model ")

    h_bias.histogram *= 1 + bias_derivative

    return h_bias


def smear_peak(energy, bins=None):
    """
    Smear a peak from a single energy using regular gaussian.

    :param energy: peak energy in keV, scalar
    :param bins: bins of the returned histogram
    :return: smeared histogram
    """
    if bins is None:
        bins = BINS

    bcs = 0.5 * (bins[1:] + bins[:-1])
    spectrum = stats.norm.pdf(bcs, energy, energy_res(energy))
    hist = Hist1d.from_histogram(spectrum, bin_edges=bins)

    return hist


def smear_hist(hist, bins=None):
    """
    Smear a histogram. This allows for non-uniform histogram binning.

    :param hist: the spectrum we want to smear
    :param bins: bin edges of the returned spectrum
    :return: smeared histogram in the same unit as input spectrum
    """
    if bins is None:
        bins = BINS

    e_true_s, rates, bin_volumes = hist.bin_centers, hist.histogram, hist.bin_volumes()
    mask = np.where(e_true_s > 0)
    e_true_s = e_true_s[mask]
    rates = rates[mask]
    bin_volumes = bin_volumes[mask]

    e_smeared_s = 0.5 * (bins[1:] + bins[:-1])
    smeared = np.zeros_like(e_smeared_s)
    for idx, e_smeared in enumerate(e_smeared_s):
        probs = stats.norm.pdf(e_smeared, loc=e_true_s, scale=energy_res(e_true_s)) * bin_volumes
        smeared[idx] = np.sum(probs * rates)

    hist_smeared = Hist1d.from_histogram(smeared, bins)

    return hist_smeared


# skew gaussian as an alternative smearing model
# https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:yue:lower:smearing_v7
# default empirical parameters are obtained with the fit in which g1 g2 from supermega are used

# numba doesn't support scipy.special, so we use np.interp to provide scipy.special.erf
# besides, numba doesn't support extrapolation, so two extreme numbers are attached
ERF_X = np.linspace(-5, 5, 1000)
ERF_X = np.concatenate([[-1e6], ERF_X, [1e6]])
ERF_Y = special.erf(ERF_X)


@numba.njit()
def interp_erf(x):
    return np.interp(x, ERF_X, ERF_Y)


@numba.njit()
def skewgauss(x, a, w, loc, N):
    t1 = 1 / (w * (2 * np.pi) ** 0.5)
    t2 = np.exp(-(x - loc) ** 2 / (2 * w ** 2))
    t3 = 1 + interp_erf(a * (x - loc) / w / 2 ** 0.5)
    return N * t1 * t2 * t3


@numba.njit()
def mean_skewgauss(loc, a, w):
    """
    Get mean value from skew gaussian.

    :param loc: location of skew gaussian
    :param a: skewness
    :param w: width
    :return: mean value
    """
    return loc + np.sqrt(2 / np.pi) * a * w / np.sqrt(1 + a ** 2)


@numba.njit()
def skew_over_E(E, amplitude=1.999, power=-1.256):
    """
    Return skewness for skew gaussian.
    Ref: xenon:xenonnt:analysis:ntsciencerun0:g1g2_update#standard_gaussian_vs_skew-gaussian_yue

    :param E: peak true energy in keV
    :return: skewness, dimensionless.
    """
    skewness = amplitude * (E ** power) * E
    return skewness


@numba.njit()
def w_over_E(E, a=37.2e-2, b=4.36e-3):
    """
    Return width for skew gaussian. When skewness is 0, width is then energy resolution for regular gaussian.
    Ref: xenon:xenonnt:analysis:ntsciencerun0:g1g2_update#standard_gaussian_vs_skew-gaussian_yue

    :param E: peak energy in keV
    :return: width , dimensionless
    """
    width = (a * E ** (-0.5) + b) * E
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
    loc = e - np.sqrt(2 / np.pi) * a * w / np.sqrt(1 + a ** 2)
    return loc


@numba.njit()
def skew_gaussian_loc(e):
    """
    Get skewed Gaussian location from the mean energy

    :param e: mean energy in keV
    :return: location of skewed Gaussian in keV
    """
    # width in kev
    w = w_over_E(e)
    # dimensionless skewness
    a = skew_over_E(e)

    loc = _skew_gaussian_loc(e, a, w)
    return loc


def get_skew_model(E, xs=None):
    a = skew_over_E(E) * E
    w = w_over_E(E) * E
    e = mean_skewgauss(a, w, E)
    if xs is None:
        xs = np.linspace(0, E * 5, 1000)
    x_center = 0.5 * (xs[1:] + xs[:-1])
    h = Hist1d(bins=xs)
    h.histogram = skewgauss(x_center, a, w, e, 1)
    return h


def smear_skew_gaussian_peak(energy, bins=None, add_bias=True):
    """
    Smear a peak from a single energy using skew gaussian.

    :param energy: peak energy in keV, scalar
    :param bins: bins of the returned histogram
    :param add_bias: boolean, include reconstruction bias in spectrum modeling if True. Default True.
    :return: smeared histogram
    """
    if bins is None:
        bins = BINS

    if add_bias:
        energy = biased_energy(energy)

    bcs = 0.5 * (bins[1:] + bins[:-1])
    skew_loc = skew_gaussian_loc(energy)
    skew_scale = w_over_E(energy)
    skewness = skew_over_E(energy)

    spectrum = stats.skewnorm.pdf(bcs, a=skewness, loc=skew_loc, scale=skew_scale)
    hist = Hist1d.from_histogram(spectrum, bin_edges=bins)

    return hist


@numba.njit()
def _smear_skew_array(e_true_s, rates, bin_volumes, bins):
    mask = np.where(e_true_s > 0)
    e_true_s = e_true_s[mask]
    rates = rates[mask]
    bin_volumes = bin_volumes[mask]

    # convert to skew gaussian params
    skew_loc_s = skew_gaussian_loc(e_true_s)
    skew_scale_s = w_over_E(e_true_s)
    skewness_s = skew_over_E(e_true_s)

    e_smeared_s = 0.5 * (bins[1:] + bins[:-1])
    smeared = np.zeros_like(e_smeared_s)

    for idx, e_smeared in enumerate(e_smeared_s):
        # probs = stats.skewnorm.pdf(e_smeared, a=skewness_s, loc=skew_loc_s, scale=skew_scale_s) * bin_volumes
        probs = skewgauss(e_smeared, a=skewness_s, loc=skew_loc_s, w=skew_scale_s, N=1) * bin_volumes
        smeared[idx] = np.sum(probs * rates)

    return smeared


def smear_skew_hist(hist, bins=None, add_bias=True):
    """
    Smear a histogram using skew gaussian. This allows for non-uniform histogram binning.

    :param hist: the spectrum we want to smear
    :param bins: bin edges of the returned spectrum
    :param add_bias: boolean, include reconstruction bias in spectrum modeling if True. Default True.
    :return: smeared histogram using skew gaussian in the same unit as input spectrum
    """
    if bins is None:
        bins = BINS

    if add_bias:
        hist = bias_hist(hist)

    e_true_s, rates, bin_volumes = hist.bin_centers, hist.histogram, hist.bin_volumes()
    smeared = _smear_skew_array(e_true_s, rates, bin_volumes, bins)

    hist_smeared = Hist1d.from_histogram(smeared, bins)
    return hist_smeared


def edges(centers):
    diff = np.diff(centers)
    return np.linspace(centers[0] - 0.5 * diff[0], centers[-1] + 0.5 * diff[-1],
                       len(centers) + 1)


# do we really need this?
def clip(hist, cut):
    """
    clip hist outside the cut region
    :param hist: original hist
    :param cut: region of the remaining hist, tuple
    :return:
    """
    # remove all bin_centers whose right/left edge is outside cutoff
    left_edges = hist.bin_edges[:-1]
    right_edges = hist.bin_edges[1:]
    i = np.where((cut[0] < right_edges) & (left_edges < cut[1]))
    bins = hist.bin_centers[i]
    h = Hist1d.from_histogram(hist.histogram[i], bin_edges=edges(bins))
    return h


# efficiency
def detection_eff(ces, return_error=False, **kwargs):
    """
    Detection efficiency as function of ces

    :param ces: ces in keV
    :return: detection efficiency
    """
    # https://xe1t-wiki.lngs.infn.it/doku.php?id=lanqing:revised_pema_found_definition
    file_name = data_path("lower/efficiency/detection_eff_s13fold_20220822_v6.csv")
    data = pd.read_csv(file_name)
    f = interpolate.interp1d(data["ER_CES"], data["MPE"], bounds_error=False,
                             fill_value=(data["MPE"].values[0], data["MPE"].values[-1]))

    if return_error:
        force_recalculate = kwargs.get("force_recalculate", False)
        if "eff_upper" not in data.columns or force_recalculate:
            print("lower and upper bounds are not in detection efficiency file, will generate\n")

            data = add_lower_upper_bound(data)

            data.to_csv(file_name)
            print("lower and upper bounds added\n")
            print(f"file saved to {file_name}")

        f_upper = interpolate.interp1d(data["ER_CES"], data["eff_upper"], bounds_error=False,
                                       fill_value=(data["eff_upper"].values[0], data["eff_upper"].values[-1]))
        f_lower = interpolate.interp1d(data["ER_CES"], data["eff_lower"], bounds_error=False,
                                       fill_value=(data["eff_lower"].values[0], data["eff_lower"].values[-1]))

        eff = f(ces)
        # use asymmetric uncertainty
        unc_lower = eff - f_lower(ces)
        unc_upper = f_upper(ces) - eff
        return eff, (unc_lower, unc_upper)
    else:
        eff = f(ces)
        return eff


def s2_threshold_acceptance(ces, threshold=500, return_error=False, **kwargs):
    """
    Return S2 threshold cut acceptance in CES space. Mostly for Rn220 CES fit and background CES fit.

    :param ces: scalar or array, in keV
    :param s2_threshod: in PE, can only be 500 (Rn220, background), or 0 (no S2 threshold).
    :param return_error: return errors if True. Default False.
    :return:
    """
    if threshold == 0:
        if return_error:
            return np.ones_like(ces), (np.zeros_like(ces), np.zeros_like(ces))
        else:
            return np.ones_like(ces)

    if threshold == 500:
        file_name = data_path("lower/efficiency/s2_threshold_500pe_acc_s13fold_20220822_v6.csv")
    else:
        raise ValueError("'threshold' can only be 0 or 500.")

    data = pd.read_csv(file_name)
    f = interpolate.interp1d(data["ER_CES"], data["MPE"], bounds_error=False,
                             fill_value=(data["MPE"].values[0], data["MPE"].values[-1]))

    if return_error:
        force_recalculate = kwargs.get("force_recalculate", False)
        if "eff_upper" not in data.columns or force_recalculate:
            print("lower and upper bounds are not in s2 threshold acceptance file, will generate\n")

            data = add_lower_upper_bound(data)

            data.to_csv(file_name)
            print("lower and upper bounds added\n")
            print(f"file saved to {file_name}")

        f_upper = interpolate.interp1d(data["ER_CES"], data["eff_upper"], bounds_error=False,
                                       fill_value=(data["eff_upper"].values[0], data["eff_upper"].values[-1]))
        f_lower = interpolate.interp1d(data["ER_CES"], data["eff_lower"], bounds_error=False,
                                       fill_value=(data["eff_lower"].values[0], data["eff_lower"].values[-1]))

        eff = f(ces)
        # use asymmetric uncertainty
        unc_lower = eff - f_lower(ces)
        unc_upper = f_upper(ces) - eff
        return eff, (unc_lower, unc_upper)
    else:
        eff = f(ces)
        return eff


def ac_cut_acceptance(ces, return_error=False):
    """
    Return acceptance loss of anti-AC cuts (shadow + ambiance). This can not be estimated by N-1 method using
    Rn220 or Ar37 data since the acceptance is rate dependent.

    ref:
      * xenon:xenonnt:analysis:ntsciencerun0:lower_efficiency_and_smearing#cut_acceptance
      FIXME: a final reference

    :param ces:
    :return:
    """
    # currently it's 93+-1 %
    eff = np.ones_like(ces) * 0.93
    if return_error:
        # symmetric
        unc = np.ones_like(ces) * 0.01
        return eff, (unc, unc)
    else:
        return eff


def ces_acc_fit(x, a, b, c, d):
    """for N-1"""
    return (a + b * x) * (1 - c * np.exp(-x / d))


def linear_fit(x, a, b):
    """Also for N-1"""
    return a + b * x


def _cut_acceptance(ces, return_error=False, cut_acp_option="final", nr_blind=True):
    """
    Get cut acceptance over CES, valid for (1, 140) keV.
    Neither S2 threshold acceptance nor AC cut acceptance is included!

    ref:
      * v1: https://github.com/XENONnT/nton/blob/10ae2d3b798476a6f7bead2f4160d5edf7e630f7/nton/lower/ceseff.py#L355-L371
      * v2: xenon:xenonnt:analysis:ntsciencerun0:lower_efficiency_and_smearing#cut_acceptance

    :param ces: scalar or array-like, in keVee
    :param return_error: return uncertainty if True. Default False.
    :param cut_acp_option: str, this impacts cut acceptance at low energies. Default "combine".
      * "avg": use the average between Rn220 and Ar37
      * "combine": use Ar37 at low energies
      * "rn220_only": use Rn220 at low energies
      * "flat": just use unity
    :return:
    """
    # rn220 + ar37
    popt_combine = [8.74887286e-01, -2.78620042e-04, -1.52828582e-01, 2.74949798e+00]
    # rn220 only
    popt_rn220 = [8.91179574e-01, -4.19438837e-04, 2.14536762e-01, 2.34874756e+00]
    # Rn220 only, after updating method to calculate S2Width
    # https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:jingqiang:xenonnt:sr0:cut_acceptance_impact_lower#how_to_properly_calculate_the_acceptance_using_rn220
    popt_rn220_update = [8.85635404e-01, -2.88848402e-04]
    # final
    popt_final = [8.87803433e-01, -3.14242018e-04, 1.06727572e-01, 2.64110418e+00]

    if cut_acp_option == "avg":
        # take average as final acceptance
        acceptance = 0.5 * (ces_acc_fit(ces, *popt_combine) + ces_acc_fit(ces, *popt_rn220))
    elif cut_acp_option == "combine":
        acceptance = 1 * (ces_acc_fit(ces, *popt_combine))
    elif cut_acp_option == "rn220_only":
        acceptance = 1 * (ces_acc_fit(ces, *popt_rn220))
    elif cut_acp_option == "flat":
        acceptance = 1 * np.ones(len(ces))
    elif cut_acp_option == "rn220_update":
        acceptance = linear_fit(ces, *popt_rn220_update)
    elif cut_acp_option == "final":
        acceptance = linear_fit(ces, *popt_final[:2])
    else:
        raise ValueError("Please choose avg, combine,rn220_only or flat")

    # artificial acceptance drop because of NR blinding
    if nr_blind:
        acceptance = (acceptance * stats.norm.cdf(2)) * (ces < 10) + acceptance * (ces >= 10)

    # safeguard for ces > 140. For ces>140, set the acceptance to the value at 140 keV
    if np.min(ces) > 140:
        acceptance = 0.8 * np.ones_like(ces)
    elif np.max(ces) > 140:
        boundary_mask = (ces > 140)
        acceptance[boundary_mask] = 0.8 * np.ones_like(acceptance[boundary_mask])
    
    if return_error:
        if cut_acp_option == "rn220_update":
            # 4% sys. unc. between ar37 and rn220
            # https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:jingqiang:xenonnt:sr0:cut_acceptance_impact_lower#how_to_properly_calculate_the_acceptance_using_rn220
            return acceptance, (0.042, 0.042)
        elif cut_acp_option == "final":
            errors = acceptance - ces_acc_fit(ces, *popt_final)
            return acceptance, (errors, errors)
        # stats unc really small, final acc dominated by ac cuts -- let's assign this unc as 0
        return acceptance, (0, 0)
    else:
        return acceptance


def cut_acceptance(ces, s2_threshold=500, return_error=False, cut_acp_option="final", nr_blind=True):
    """
    Cut acceptance in LowER. Both S2 threshold and ac cut acceptance are included.

    :param ces: scalar or array, in keV
    :param s2_threshod: in PE, can only be 500 (Rn220, background),
    650 (previous background), or 0 (no S2 threshold).
    :param return_error: return uncertainty if True. Default False.
    :return:
    """
    cut_acc = _cut_acceptance(ces, return_error=return_error, cut_acp_option=cut_acp_option, nr_blind=nr_blind)
    s2_acc = s2_threshold_acceptance(ces, threshold=s2_threshold, return_error=return_error)
    ac_acc = ac_cut_acceptance(ces, return_error=return_error)

    if return_error:
        acc = cut_acc[0] * s2_acc[0] * ac_acc[0]
        # the uncertaint propation follows the rule of production
        # relative uncertainty
        unc_lower = np.sqrt((cut_acc[1][0] / cut_acc[0]) ** 2
                            # just in case s2_acc is 0
                            + (s2_acc[1][0] / np.maximum(s2_acc[0], 1e-8)) ** 2
                            + (ac_acc[1][0] / ac_acc[0]) ** 2)
        unc_upper = np.sqrt((cut_acc[1][1] / cut_acc[0]) ** 2
                            # just in case s2_acc is 0
                            + (s2_acc[1][1] / np.maximum(s2_acc[0], 1e-8)) ** 2
                            + (ac_acc[1][1] / ac_acc[0]) ** 2)
        # absolute uncertainty
        # if acc is 0, then uncertainty becomes 0 -- also makes sense
        unc_lower *= acc
        unc_upper *= acc
        return acc, (unc_lower, unc_upper)

    else:
        acc = cut_acc * s2_acc * ac_acc
        return acc


def efficiency(ces, s2_threshold=500, return_error=False, cut_acp_option="final", nr_blind=True):
    """
    Total efficiency (detection + selection).

    :param ces: in keV
    :param s2_threshod: in PE, can only be 500 (Rn220, background), or 0 (no S2 threshold).
    :param return_error: return error if True. Default False.
    :return:
    """
    det_eff = detection_eff(ces, return_error=return_error)
    cut_acc = cut_acceptance(ces, s2_threshold=s2_threshold, return_error=return_error, cut_acp_option=cut_acp_option,
                             nr_blind=nr_blind)

    if return_error:
        eff = det_eff[0] * cut_acc[0]
        # relative
        # in case detection efficiency and/or cut acceptance is 0
        unc_lower = np.sqrt((det_eff[1][0] / np.maximum(det_eff[0], 1e-8)) ** 2
                            + (cut_acc[1][0] / np.maximum(cut_acc[0], 1e-8)) ** 2)
        unc_upper = np.sqrt((det_eff[1][1] / np.maximum(det_eff[0], 1e-8)) ** 2 +
                            (cut_acc[1][1] / np.maximum(cut_acc[0], 1e-8)) ** 2)
        # absolute
        unc_lower *= eff
        unc_upper *= eff
        return eff, (unc_lower, unc_upper)
    else:
        eff = det_eff * cut_acc
        return eff


def shape_weight(ces, emin=10, emax=20):
    """
    Weight for the shape parameter. The idea is to restrict the impact of shape parameter at low energies so that it can
    focus on near-threshold effect. The weight will be reduced to 0 at high energies in a continuous way.

    :param ces: scalar or array like, in keV
    :param emin: scalar, in keV. The minimum energy to begin weight reducing. Before emin, weight is 1.
    :param emax: scalar, in keV. The maximum energy to end weight reducing. After emax, weight is 0.
    :return:
    """
    return 1 * (ces <= emin) + (emax - ces) / (emax - emin) * ((ces > emin) & (ces < emax)) + 0 * (ces > emax)


def shaped_efficiency(ces, a=0, s2_threshold=500, cut_acp_option="final", nr_blind=True):
    """
    Efficiency with a shape parameter.

    :param ces: in keV.
    :param a: shape parameter that should be within (-1, 1). Default 0.
    :param s2_threshold: in PE, can only be 500 (Rn220), or 0 (no S2 threshold).
    :return:
    """
    eff, (unc_lower, unc_upper) = efficiency(ces, s2_threshold=s2_threshold,
                                             return_error=True, cut_acp_option=cut_acp_option, nr_blind=nr_blind)
    eff += shape_weight(ces) * ((a * unc_lower) * (a < 0) + (a * unc_upper) * (a > 0))
    return eff


def apply_efficiency(hist, a=0, s2_threshold=500, cut_acp_option="final", nr_blind=True):
    """
    Takes a multihist object, applies efficiency to it, and returns it as a new multihist

    :param hist: smeared hist
    :param a: shape parameter within (-1, 1). Default 0.
    :param s2_threshold: in PE, can only be 500 (Rn220), 650 (background), or 0 (no S2 threshold). Default 650.
    :return:
    """
    newhist = deepcopy(hist)
    newhist = newhist * shaped_efficiency(newhist.bin_centers, a=a, s2_threshold=s2_threshold,
                                          cut_acp_option=cut_acp_option, nr_blind=nr_blind)
    return newhist


def deapply_efficiency(hist, **kwargs):
    """
    De-apply efficiency to a hist to get the 'true' (0-efficiency-loss) histogram.
    Doesn't work for region where efficiency is 0.

    :param hist:

    :return:
    """
    newhist = deepcopy(hist)
    eff = shaped_efficiency(newhist.bin_centers, **kwargs)
    newhist = newhist / np.where(eff == 0, 1, eff)
    return newhist


def add_lower_upper_bound(data):
    """
    Add lower and upper 1-sigma bound of efficiency to data

    :param data: detection efficiency/s2 threshold acceptance panda dataframe from BBF
    :return: data with two more columns, 'eff_upper' and 'eff_lower'
    """
    lower_bound = []
    upper_bound = []
    # hardcoded, 100 energies
    for i in range(200):
        toymc = data.loc[i].values[3:103]
        mpe = data.loc[i, "MPE"]
        toymc_upper = toymc[toymc >= mpe]
        toymc_lower = toymc[toymc <= mpe]

        _lower_bound = mpe
        _upper_bound = mpe

        if len(toymc_lower):
            _lower_bound = np.percentile(toymc_lower, q=100 - 68.3)
        if len(toymc_upper):
            _upper_bound = np.percentile(toymc_upper, q=68.3)

        lower_bound.append(_lower_bound)
        upper_bound.append(_upper_bound)

    data["eff_upper"] = upper_bound
    data["eff_lower"] = lower_bound
    return data
