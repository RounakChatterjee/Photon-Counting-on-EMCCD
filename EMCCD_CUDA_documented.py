"""
EMCCD photon-counting and correlation analysis utilities.

This module provides helper functions for:
    1. Reading EMCCD data from NumPy binary files or SIF files.
    2. Building measured probability distributions from camera counts.
    3. Fitting EMCCD noise and photon-counting models.
    4. Estimating photon-number thresholds.
    5. Applying thresholds to EMCCD frames.
    6. Computing frame correlations/convolutions, optionally on GPU using CuPy.

Notes
-----
- The model follows an EMCCD multiplication-register picture with:
    * Gaussian readout noise,
    * clock-induced charge (CIC),
    * serial-register noise,
    * photon-generated multiplication statistics.
- `bProb` is a global beam-probability map. It is assigned using `Beam(data, assign_prob=True)`
  and then used by the photon model and threshold-frame generation.
- GPU acceleration is used for the photon distribution and correlation functions when CuPy is available.

Author: Rounak, 2026
"""

# =============================================================================
# Imports
# =============================================================================

import json
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import gc

import numpy as np
import matplotlib.pyplot as plt

from numpy import convolve
from scipy.stats import norm, erlang, poisson
from scipy.signal import fftconvolve as scipy_fftconvolve
from lmfit import Model

import cupy as cp
from cupyx.scipy.special import i1
from cupyx.scipy.signal import fftconvolve as cupy_fftconvolve

import sif_parser as pr


# =============================================================================
# Global configuration
# =============================================================================

# Global beam-probability distribution. Assigned using Beam(..., assign_prob=True).
bProb = np.array([])

# If True, fitting functions print parameters during every model evaluation.
echo_flag = True

# Number of EMCCD multiplication registers.
Nregisters = 552

# Continuous count axis used internally for convolution-based probability models.
x_cont = np.arange(-5000, 5001, 1)


# =============================================================================
# Data I/O
# =============================================================================

def readBinary(
    path: str = "",
    file: str = "",
    frames: int = 0,
    frame_size: Sequence[int] = (512, 512),
) -> np.ndarray:
    """
    Read an EMCCD dataset stored as a NumPy binary file.

    Parameters
    ----------
    path : str
        Directory path containing the file.
    file : str
        File name, usually ending in `.npy`.
    frames : int, optional
        Number of frames to return. If 0, all frames are returned.
    frame_size : sequence of int, optional
        Expected frame size. This argument is kept for compatibility but is not
        used directly because NumPy stores the array shape internally.

    Returns
    -------
    np.ndarray
        Data array with shape `(frames, rows, columns)` and dtype `int32`.
    """
    print("reading:" + path + file, end="\t")
    data = np.load(path + f"{file}").astype(np.int32)
    print("File Read!")

    if frames != 0:
        return data[:frames, :, :]

    return data


def readSif(
    path: str = "",
    file: str = "",
    frames: int = 0,
) -> Tuple[np.ndarray, object]:
    """
    Read an Andor SIF file.

    Parameters
    ----------
    path : str
        Directory path containing the SIF file.
    file : str
        SIF file name.
    frames : int, optional
        Number of frames to return. If 0, all frames are returned.

    Returns
    -------
    data : np.ndarray
        EMCCD frames.
    info : object
        Metadata returned by `sif_parser.np_open`.
    """
    print("reading:" + path + file, end="\t")
    data, info = pr.np_open(path + file)
    print("File Read!")

    if frames != 0:
        return data[:frames], info

    return data, info


# =============================================================================
# Histogram / probability distribution utilities
# =============================================================================

def make_prob(
    data: np.ndarray,
    range: Sequence[int] = (400, 2500),
    show_plot: bool = False,
    log_plot: bool = True,
    make_offset: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert raw camera data into a normalized count probability distribution.

    The histogram is computed from all pixels in all frames. Optionally, the x-axis
    is shifted so that the most probable count value becomes zero.

    Parameters
    ----------
    data : np.ndarray
        EMCCD data array.
    range : sequence of int, optional
        Histogram count range `[min_count, max_count]`.
    show_plot : bool, optional
        If True, plot the probability distribution.
    log_plot : bool, optional
        If True, use logarithmic y-axis for the plot.
    make_offset : bool, optional
        If True, subtract the peak-count location from the count axis.

    Returns
    -------
    x : np.ndarray
        Count axis, optionally offset-corrected.
    px : np.ndarray
        Normalized sample probability.
    offset_x : float
        Count value corresponding to the maximum of the histogram.
    """
    hist = np.histogram(
        data.flatten(),
        range=range,
        bins=np.diff(range)[0],
        density=True,
    )

    px = hist[0]
    x = hist[1][:-1]

    offset_x = x[np.argmax(px)]

    if make_offset:
        x = x - offset_x

    if show_plot:
        plt.scatter(x, px, label=f"Offset = {offset_x}")
        plt.xlabel("Counts", size=15)
        plt.ylabel("Sample Probability", size=15)
        plt.legend(fontsize=15)
        if log_plot:
            plt.yscale("log")
        plt.show()

    return x, px, offset_x


def weight(prob: np.ndarray, precision: int = 5) -> np.ndarray:
    """
    Generate fitting weights based on the order of magnitude of probability values.

    This is useful when fitting a probability distribution on a log scale, where the
    low-probability tail should not be completely ignored by least-squares fitting.

    Parameters
    ----------
    prob : np.ndarray
        Probability values.
    precision : int, optional
        Maximum order-of-magnitude weighting. Values below `10^-precision` receive
        weight `10^precision`.

    Returns
    -------
    np.ndarray
        Weight array of the same length as `prob`.
    """
    weights = np.ones(len(prob))

    for i in range(precision):
        mask = (prob > 10 ** (-i - 1)) & (prob <= 10 ** (-i))
        weights[mask] = (10 ** i) * weights[mask]

    weights[prob <= 10 ** (-precision)] = 10 ** precision

    return weights

def weight_function(prob: np.ndarray, precision: int = 5) -> np.ndarray:
    """
    This funciton assigns the weight based on the original formula of the weight 
    funciton giben as W(x) = 10^(floor(|log10(P(x))|))
    where P(x) is the probability of EMCCD count x and W(x) is the assigned weight

    This is useful when fitting a probability distribution on a log scale, where the
    low-probability tail should not be completely ignored by least-squares fitting.
    Parameters
    ----------
    prob : np.ndarray
        Probability values.
    precision : int, optional
        Maximum order-of-magnitude weighting. Values below `10^-precision` receive
        weight `10^precision`.

    Returns
    -------
    np.ndarray
        Weight array of the same length as `prob`.
    """
    weights = np.ones(len(prob))
    
    weights[prob<=np.pow(10.0,-precision)] = np.pow(10.0,precision)
    
    weights[prob>np.pow(10.0,-precision)] = np.pow(10.0,np.floor(np.abs(np.log10(prob[prob>=np.pow(10.0,-precision)]))))
    
    return weights

# =============================================================================
# EMCCD probability model
# =============================================================================

def Erlang(x: np.ndarray, n: int, G: float) -> np.ndarray:
    """
    Erlang distribution used for EMCCD multiplication statistics.

    Parameters
    ----------
    x : np.ndarray
        Count axis.
    n : int
        Shape parameter, often interpreted as photon/electron number.
    G : float
        EMCCD gain scale.

    Returns
    -------
    np.ndarray
        Erlang probability density evaluated on `x`.
    """
    val = erlang.pdf(x, n, 0, G)
    val[x == 0] = 0
    return val


def p_noise(
    x: np.ndarray,
    sigma: float,
    pCIC: float,
    pser: float,
    pc: float,
) -> np.ndarray:
    """
    EMCCD noise probability model.

    The total noise distribution is modeled as the convolution of:
        1. Gaussian readout noise,
        2. clock-induced charge contribution,
        3. serial-register noise contribution.

    Parameters
    ----------
    x : np.ndarray
        Count axis on which the final model should be returned.
    sigma : float
        Standard deviation of Gaussian readout noise.
    pCIC : float
        Clock-induced charge probability.
    pser : float
        Serial-register noise probability per register.
    pc : float
        Multiplication probability per register.

    Returns
    -------
    np.ndarray
        Normalized noise probability on the requested count axis `x`.
    """
    if echo_flag:
        print(sigma, pCIC, pser, pc)

    G = (1 + pc) ** Nregisters

    # Gaussian read noise.
    c1 = norm.pdf(x_cont, 0, sigma)

    # Clock-induced charge contribution.
    c2 = pCIC * Erlang(x_cont, 1, G)
    c2[x_cont == 0] = 1 - pCIC

    # Serial-register contribution.
    c3 = np.zeros(len(x_cont))
    for k in range(1, Nregisters + 1):
        c3 += pser * Erlang(x_cont, 1, (1 + pc) ** (Nregisters - k))
    c3[x_cont == 0] = 1.0 - Nregisters * pser

    # Total noise distribution.
    val = convolve(c1, convolve(c2, c3, mode="same"), mode="same")

    indices = np.where((x_cont >= x[0]) & (x_cont <= x[-1]))
    return val[indices] / np.sum(val[indices])


def p_photon(
    x: np.ndarray,
    pc: float,
    mu: np.ndarray,
) -> np.ndarray:
    """
    Photon-generated EMCCD output probability model.

    This model uses the analytical EMCCD multiplication distribution for a Poisson
    input photon/electron number with mean `mu`.

    Parameters
    ----------
    x : np.ndarray
        Count axis.
    pc : float
        Multiplication probability per EMCCD register.
    mu : np.ndarray
        Mean photon number(s). If multiple values are provided, the returned
        distribution is averaged over all values.

    Returns
    -------
    np.ndarray
        Photon probability distribution on `x`.
    """
    x_positive = cp.asarray(x[x > 0])
    mu_gpu = cp.asarray(mu)

    X, Mu = cp.meshgrid(x_positive, mu_gpu)
    G = (1 + pc) ** Nregisters

    val = cp.exp(-Mu - X / G) * cp.sqrt(Mu / (G * X)) * i1(
        2 * cp.sqrt((Mu * X / G))
    )

    # Add probability at x = 0.
    zero_column = cp.array([cp.exp(-mu_gpu)]).T
    val = cp.hstack((zero_column, val))

    # Pad negative x side with zeros.
    val = cp.pad(val, ((0, 0), (len(x[x < 0]), 0)))

    val = cp.nan_to_num(val)

    if len(mu) != 1:
        return cp.asnumpy(cp.mean(val, axis=0))

    return cp.asnumpy(val[0])


def p_combined(
    x: np.ndarray,
    mu: float,
    sigma: float,
    pCIC: float,
    pser: float,
    pc: float,
) -> np.ndarray:
    """
    Combined EMCCD model including noise and photon contribution.

    Parameters
    ----------
    x : np.ndarray
        Count axis on which the model should be returned.
    mu : float
        Total mean photon number scale. Pixel-wise mean values are calculated using
        `mu * bProb`.
    sigma : float
        Gaussian read-noise standard deviation.
    pCIC : float
        Clock-induced charge probability.
    pser : float
        Serial-register noise probability.
    pc : float
        Multiplication probability per register.

    Returns
    -------
    np.ndarray
        Normalized probability distribution on count axis `x`.
    """
    if bProb.size == 0:
        raise ValueError("bProb is empty. Run Beam(data, assign_prob=True) before fitting photons.")

    mu_list = (mu * bProb).flatten()

    noise_vals = p_noise(x_cont, sigma, pCIC, pser, pc)
    photon_vals = p_photon(x_cont, pc, mu_list)

    val = convolve(noise_vals, photon_vals, mode="same")

    indices = np.where((x_cont >= x[0]) & (x_cont <= x[-1]))
    return_val = val[indices] / np.sum(val[indices])

    if echo_flag:
        print(mu, sigma, pCIC, pser, pc)
        print(
            "\n Noise Prob = ",
            np.sum(noise_vals),
            " ,Photon Prob = ",
            np.sum(photon_vals),
            ", Total = ",
            np.sum(val),
        )
        plt.plot(x, return_val)
        plt.yscale("log")
        plt.show()

    return return_val


# =============================================================================
# Beam and background correction utilities
# =============================================================================

def Beam(data: np.ndarray, assign_prob: bool = False) -> None:
    """
    Display the average beam image and optionally assign it as a probability map.

    Parameters
    ----------
    data : np.ndarray
        Either a 3D frame stack `(frames, rows, cols)` or a 2D image.
    assign_prob : bool, optional
        If True, negative values are clipped to zero and the normalized beam profile
        is stored globally as `bProb`.
    """
    global bProb

    if len(data.shape) == 3:
        beam = np.mean(data, axis=0)
    else:
        beam = data.copy()

    if assign_prob:
        beam[beam < 0] = 0
        bProb = beam / np.sum(beam)

        plt.imshow(bProb)
        plt.colorbar()
        plt.title("Assigned beam probability map")
        plt.show()
    else:
        plt.imshow(beam)
        plt.colorbar()
        plt.title("Average beam")
        plt.show()


def Bg_Correction(
    data: np.ndarray,
    correction: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply row-column background correction to EMCCD frames.

    If no correction matrix is supplied, the correction matrix is estimated from the
    input data. In that case, the input data should ideally be a noise/background-only
    dataset.

    Correction model:
        Correction(row, col) = row_mean(row) + col_mean(col) - global_mean

    Parameters
    ----------
    data : np.ndarray
        Frame stack with shape `(frames, rows, cols)`.
    correction : np.ndarray or None, optional
        Precomputed correction matrix. If None, it is estimated from `data`.

    Returns
    -------
    corrected_data : np.ndarray
        Background-corrected data.
    correction : np.ndarray
        Returned only when correction is estimated internally.
    """
    corrected_data = np.zeros(data.shape, dtype=np.float32)

    if correction is None:
        row_vals = np.mean(data, axis=(0, 2))
        col_vals = np.mean(data, axis=(0, 1))
        img_mean = np.mean(data)

        R, C = np.meshgrid(col_vals, row_vals)
        B = img_mean * np.ones(shape=(len(row_vals), len(col_vals)))

        correction_matrix = R + C - B
        corrected_data[:] = data[:] - correction_matrix

        return corrected_data, correction_matrix

    corrected_data[:] = data[:] - correction
    return corrected_data


# =============================================================================
# Fitting functions
# =============================================================================

def Noisefit(
    px: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
    params: Sequence[float],
    vary_flags: Optional[Dict[str, bool]] = None,
    method: str = "leastsq",
):
    """
    Fit the EMCCD noise model to a measured probability distribution.

    Parameters
    ----------
    px : np.ndarray
        Measured probability distribution.
    x : np.ndarray
        Count axis corresponding to `px`.
    weights : np.ndarray
        Fitting weights.
    params : sequence of float
        Initial values `[sigma, pCIC, pser, pc]`.
    vary_flags : dict, optional
        Dictionary controlling which parameters vary during fitting.
    method : str, optional
        lmfit minimization method.

    Returns
    -------
    lmfit.model.ModelResult
        Fit result object.
    """
    if vary_flags is None:
        vary_flags = {"sigma": True, "pCIC": True, "pser": True, "pc": True}

    sigma, pCIC, pser, pc = params

    fit_model = Model(p_noise, nan_policy="omit")

    fit_model.set_param_hint("sigma", value=sigma, min=0, max=50.0, vary=vary_flags["sigma"])
    fit_model.set_param_hint("pCIC", value=pCIC, min=0, max=1, vary=vary_flags["pCIC"])
    fit_model.set_param_hint("pser", value=pser, min=0, max=1, vary=vary_flags["pser"])
    fit_model.set_param_hint("pc", value=pc, min=0, max=1, vary=vary_flags["pc"])

    result = fit_model.fit(px, x=x, weights=weights, method=method)
    return result


def Photonfit(
    px: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
    params: Sequence[float],
    vary_flags: Optional[Dict[str, bool]] = None,
    method: str = "leastsq",
):
    """
    Fit the combined noise + photon EMCCD model.

    Parameters
    ----------
    px : np.ndarray
        Measured probability distribution.
    x : np.ndarray
        Count axis corresponding to `px`.
    weights : np.ndarray
        Fitting weights.
    params : sequence of float
        Initial values `[mu, sigma, pCIC, pser, pc]`.
    vary_flags : dict, optional
        Dictionary controlling which parameters vary during fitting.
    method : str, optional
        lmfit minimization method.

    Returns
    -------
    lmfit.model.ModelResult
        Fit result object.
    """
    if vary_flags is None:
        vary_flags = {
            "mu": True,
            "sigma": True,
            "pCIC": True,
            "pser": True,
            "pc": True,
        }

    mu, sigma, pCIC, pser, pc = params

    fit_model = Model(p_combined, nan_policy="omit")

    fit_model.set_param_hint("mu", value=mu, min=0, max=1e6, vary=vary_flags["mu"])
    fit_model.set_param_hint("sigma", value=sigma, min=0, max=30.0, vary=vary_flags["sigma"])
    fit_model.set_param_hint("pCIC", value=pCIC, min=0, max=1, vary=vary_flags["pCIC"])
    fit_model.set_param_hint("pser", value=pser, min=0, max=1, vary=vary_flags["pser"])
    fit_model.set_param_hint("pc", value=pc, min=0, max=1, vary=vary_flags["pc"])

    result = fit_model.fit(px, x=x, weights=weights, method=method)
    return result


# =============================================================================
# Plotting and parameter handling
# =============================================================================

def plot_fit(
    x: np.ndarray,
    result,
    prob: np.ndarray,
    log_plot: bool = True,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    save_options: Optional[Dict[str, bool]] = None,
    save_file_name: str = "",
) -> Dict[str, float]:
    """
    Plot measured probability distribution and lmfit best fit.

    Parameters
    ----------
    x : np.ndarray
        Count axis.
    result : lmfit.model.ModelResult
        Fit result returned by `Noisefit` or `Photonfit`.
    prob : np.ndarray
        Measured probability distribution.
    log_plot : bool, optional
        If True, use logarithmic y-axis.
    xlim : sequence of float, optional
        x-axis limits.
    ylim : sequence of float, optional
        y-axis limits.
    save_options : dict, optional
        Example: `{"fig": True, "parameters": True}`.
    save_file_name : str, optional
        Base file name for saving figure and parameters.

    Returns
    -------
    dict
        Best-fit parameter values.
    """
    if xlim is None:
        xlim = []
    if ylim is None:
        ylim = []
    if save_options is None:
        save_options = {"fig": False, "parameters": False}

    plt.scatter(x, prob, color="red", marker="x", label="EMCCD Data")
    plt.plot(x, result.best_fit, lw=3, label="Fit", color="#00FF00")
    plt.xlabel("Counts", size=15)
    plt.ylabel("Probability", size=15)
    plt.legend(fontsize=15)

    if log_plot:
        plt.yscale("log")
    if len(xlim) != 0:
        plt.xlim(xlim)
    if len(ylim) != 0:
        plt.ylim(ylim)
    if save_options.get("fig", False):
        plt.savefig(save_file_name + ".png", dpi=1600)

    plt.show()

    params = result.best_values

    print("Data max = ", x[np.argmax(prob)], "Fit max = ", x[np.argmax(result.best_fit)])
    print(f"{result.fit_report()}")

    if save_options.get("parameters", False):
        with open(save_file_name + "_params.json", "w") as fp:
            json.dump(result.best_values, fp)

    return params


def plot_fit_data(
    x: np.ndarray,
    prob: np.ndarray,
    params: Sequence[float],
    log_plot: bool = True,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
) -> None:
    """
    Plot measured probability and model generated from supplied parameters.

    Parameters
    ----------
    x : np.ndarray
        Count axis.
    prob : np.ndarray
        Measured probability.
    params : sequence of float
        Parameters for `p_combined`: `[mu, sigma, pCIC, pser, pc]`.
    log_plot : bool, optional
        If True, use logarithmic y-axis.
    xlim : sequence of float, optional
        x-axis limits.
    ylim : sequence of float, optional
        y-axis limits.
    """
    if xlim is None:
        xlim = []
    if ylim is None:
        ylim = []

    plt.scatter(x, prob, color="red", marker="x", label="EMCCD Data")

    fit_x = p_combined(x, *params)
    plt.plot(x, fit_x, lw=3, label="Fit", color="#00FF00")

    plt.xlabel("Counts", size=15)
    plt.ylabel("Probability", size=15)
    plt.legend(fontsize=15)

    if log_plot:
        plt.yscale("log")
    if len(xlim) != 0:
        plt.xlim(xlim)
    if len(ylim) != 0:
        plt.ylim(ylim)

    plt.show()


def load_params(path: str, file: str) -> np.ndarray:
    """
    Load saved fit parameters from a JSON file.

    Parameters
    ----------
    path : str
        Directory path.
    file : str
        JSON file name.

    Returns
    -------
    np.ndarray
        Parameter values as float64 array, preserving JSON item order.
    """
    return np.float64(np.array(list(json.load(open(path + file)).items()))[:, 1])


# =============================================================================
# Photon threshold calculation
# =============================================================================

def Photon_Thresh(N: int, params: Sequence[float]) -> np.ndarray:
    """
    Estimate count thresholds separating photon-number probabilities.

    For each photon number transition, this function finds the count value at which
    the posterior probability for `n` photons becomes larger than that of `n-1`
    photons.

    Parameters
    ----------
    N : int
        Maximum photon number threshold to calculate.
    params : sequence of float
        Parameter list `[mu, sigma, pCIC, pser, pc]`.

    Returns
    -------
    np.ndarray
        Threshold values for photon numbers 1 to N.
    """
    mu = params[0]
    gain = (1 + params[-1]) ** Nregisters

    pNx = p_noise(x_cont, *params[1:])
    px = p_photon(x=x_cont, pc=params[-1], mu=np.array([mu]))
    pxx = np.convolve(pNx, px, mode="same")

    val = np.zeros(N)
    p_prev = np.zeros(len(x_cont))

    for n in range(N + 1):
        if n == 0:
            p0mu = poisson.pmf(n, mu)
            p0x = (p0mu * pNx) / pxx
            p0x[np.isnan(p0x)] = 0
            p_prev = p0x
        else:
            pxn = Erlang(x_cont, n, gain)
            pnmu = poisson.pmf(n, mu)
            pnx = np.convolve(pNx, pxn * pnmu, mode="same") / pxx
            pnx[np.isnan(pnx)] = 0

            crossing_indices = np.where((pnx - p_prev) > 0)[0]
            if len(crossing_indices) == 0:
                val[n - 1] = np.nan
            else:
                val[n - 1] = x_cont[crossing_indices[0]]

            p_prev = pnx

    return val


def Thresholder(
    data: np.ndarray,
    T_frame: np.ndarray,
    max_photon_number: int = 1,
    dtype=np.int32,
) -> np.ndarray:
    """
    Convert EMCCD count frames into photon-number-counted frames.

    Parameters
    ----------
    data : np.ndarray
        EMCCD count data with shape `(frames, rows, cols)`.
    T_frame : np.ndarray
        Threshold frame with shape `(rows, cols, number_of_thresholds)`.
    max_photon_number : int, optional
        Maximum photon number to count.
    dtype : data type, optional
        Output data type.

    Returns
    -------
    np.ndarray
        Photon-counted data with same spatial and temporal shape as `data`.
    """
    if max_photon_number > len(T_frame[0, 0, :]):
        print(
            "Not enough Threshold provided, reverting to maximum available = ",
            len(T_frame[0, 0, :]),
        )
        max_photon_number = len(T_frame[0, 0, :])

    counted_data = np.zeros(data.shape, dtype=dtype)

    for n in range(max_photon_number):
        counted_data[:, :, :] += dtype(data[:, :, :] >= T_frame[:, :, n])

    return counted_data


def create_T_frame(
    m_vals: Sequence[float],
    params: Sequence[float],
    frame_size: Sequence[int] = (200, 200),
) -> np.ndarray:
    """
    Create a pixel-wise threshold frame from a beam probability map.

    The local mean photon number of each pixel is estimated as:

        local_mu = total_mu * bProb[pixel]

    Then the nearest threshold table is chosen from the supplied `m_vals`.

    Parameters
    ----------
    m_vals : sequence of float
        Mean photon values for which threshold tables will be precomputed.
    params : sequence of float
        Parameter list `[mu, sigma, pCIC, pser, pc]`.
    frame_size : sequence of int, optional
        Spatial frame size `(rows, cols)`.

    Returns
    -------
    np.ndarray
        Threshold frame with shape `(rows, cols, number_of_thresholds)`.
    """
    if bProb.size == 0:
        raise ValueError("bProb is empty. Run Beam(data, assign_prob=True) before creating thresholds.")

    print(f"Mean Photon per frame per pixel: {params[0] * np.mean(bProb)}")
    print(f"params = {params[1:]}")
    print("Number of mean values:", len(m_vals))

    T_vals = []
    pixel_params = np.append(0, params[1:])

    for m in m_vals:
        pixel_params[0] = m
        max_threshold_number = int(m + 1.5 * np.ceil(np.sqrt(m)))
        T = Photon_Thresh(max_threshold_number, pixel_params)
        T_vals.append(T)

    # Pad all threshold arrays to the same length.
    max_len = len(T_vals[-1])
    T_mat = np.zeros(shape=(len(T_vals), max_len))

    for i, T in enumerate(T_vals):
        T_mat[i] = np.append(T, 5000 * np.ones(max_len - len(T)))

    mu_vals = params[0] * bProb.flatten()
    indices = np.searchsorted(m_vals, mu_vals)

    # Prevent out-of-range indexing if mu_vals exceed the supplied m_vals range.
    indices = np.clip(indices, 0, len(m_vals) - 1)

    T_frame = T_mat[indices].reshape(frame_size[0], frame_size[1], max_len)
    return T_frame


# =============================================================================
# Correlation / convolution utilities
# =============================================================================

def corr(
    frames: np.ndarray,
    flip_later: bool = False,
    shift: int = 1,
    gpu_on: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute frame-wise correlation or convolution with periodic boundary padding.

    This function is usually applied to photon-counted EMCCD frames.

    Interpretation
    --------------
    - If `flip_later=True`, the function computes a correlation-like quantity by
      convolving the padded frame with the flipped frame.
    - If `flip_later=False`, the function computes a convolution-like quantity.
    - `bg` is computed using a shifted frame index `i - shift`, which gives an
      accidental/background estimate.

    Parameters
    ----------
    frames : np.ndarray
        Input frame stack with shape `(N, rows, cols)`.
    flip_later : bool, optional
        If True, flip the second frame before convolution.
    shift : int, optional
        Frame shift used for background/accidental estimate.
    gpu_on : bool, optional
        If True, use CuPy GPU arrays and CuPy fftconvolve.

    Returns
    -------
    con : np.ndarray
        Accumulated signal correlation/convolution map.
    bg : np.ndarray
        Accumulated shifted-frame background map.
    """
    N, xl, yl = frames.shape

    # Enforce odd spatial dimensions by removing the first row/column if needed.
    # This gives a well-defined central pixel in the correlation output.
    if xl % 2 == 0:
        frames = frames[:, 1:]
    if yl % 2 == 0:
        frames = frames[:, :, 1:]

    N, xl, yl = frames.shape

    if gpu_on:
        con = cp.zeros((xl + 1, yl + 1))
        bg = cp.zeros((xl + 1, yl + 1))
    else:
        con = np.zeros((xl + 1, yl + 1))
        bg = np.zeros((xl + 1, yl + 1))

    progress_step = max(N // 10, 1)

    for i in range(N):
        # Periodic boundary padding.
        pad_width = ((xl // 2, xl - xl // 2), (yl // 2, yl - yl // 2))

        if gpu_on:
            frame_i = cp.asarray(frames[i])
            frame_shifted = cp.asarray(frames[i - shift])
            padded = cp.pad(frame_i, pad_width, mode="wrap")

            if flip_later:
                con += cupy_fftconvolve(padded, cp.flip(frame_i), mode="valid")
                bg += cupy_fftconvolve(padded, cp.flip(frame_shifted), mode="valid")
            else:
                con += cupy_fftconvolve(padded, frame_i, mode="valid")
                bg += cupy_fftconvolve(padded, frame_shifted, mode="valid")

        else:
            frame_i = frames[i]
            frame_shifted = frames[i - shift]
            padded = np.pad(frame_i, pad_width, mode="wrap")

            if flip_later:
                con += scipy_fftconvolve(padded, np.flip(frame_i), mode="valid")
                bg += scipy_fftconvolve(padded, np.flip(frame_shifted), mode="valid")
            else:
                con += scipy_fftconvolve(padded, frame_i, mode="valid")
                bg += scipy_fftconvolve(padded, frame_shifted, mode="valid")

        if (i + 1) % progress_step == 0:
            print("=", end="")

    print()

    # Remove the dominating auto-correlation peak if correlation mode is used.
    if flip_later:
        con[np.where(con == np.amax(con))] = 0
        bg[np.where(bg == np.amax(bg))] = 0

    if gpu_on:
        return cp.asnumpy(con), cp.asnumpy(bg)

    return con, bg


# =============================================================================
# Photon Counting and JPD utility functions
# =============================================================================

def read_photons(
    file_name = "",
    Data_path = "",
    correction=[],
    max_photon_number=1,
    dtype=np.int32,
    crop=[],
    T_frame=[]
):
    """
    Read EMCCD photon-exposed data and convert it into photon-counted frames.

    This function reads either a binary `.npy` file or a `.sif` EMCCD file,
    optionally applies background/bias correction, optionally crops the data,
    and finally thresholds the frames to obtain probabilistic photon-counted
    data.

    Parameters
    ----------
    file_name : str
        Name of the input file.

        If the file extension is `.npy`, the data is read using
        `emccd.readBinary`.

        Otherwise, the data is assumed to be a `.sif` file and is read using
        `emccd.readSif`.

        defaults to "".
    
    Data_path : str
        Name of input path

        It's useful when you have to track seperate files in seperate locations.

        defaults to "".

    correction : array-like, optional
        Background/bias correction frame.

        If provided, the raw EMCCD data is corrected using
        `emccd.Bg_Correction`.

        Default is an empty list, meaning no correction is applied.

    max_photon_number : int, optional
        Maximum photon number allowed during thresholding.

        This is passed to `emccd.Thresholder`.

        Default is `1`.

    dtype : numpy dtype, optional
        Data type of the output photon-counted array.

        Default is `np.int32`.

    crop : list or tuple, optional
        Cropping indices for selecting a region of interest.

        Expected format is:

        ```python
        crop = [row_start, row_end, col_start, col_end]
        ```

        If provided, both the data and threshold frame are cropped.

        Default is an empty list, meaning no cropping is applied.

    T_frame : array-like, optional
        Threshold frame used for photon counting.

        This should have the same spatial dimensions as the EMCCD data 
        or cropped data used.
        If crop is applied , it crops it to the same dimensions 
        as the cropped region of the data.

        Default is an empty list.

    Returns
    -------
    cd : numpy.ndarray
        Photon-counted data returned by `emccd.Thresholder`.

        The output shape depends on the input data shape and the thresholding
        settings.

    The function also manually deletes the raw data array and calls
    `gc.collect()` after thresholding to free memory.

    Example
    -------
    ```python
    photons = read_photons(
        file_name="D235.sif",
        Data_path = "Data/"
        correction=correction,
        max_photon_number=1,
        dtype=np.int32,
        crop=[100, 400, 100, 400],
        T_frame=T_frame
    )
    ```
    """

    cd = []

    # Check whether the input file is a binary NumPy file
    bin_file = True if file_name[-3:] == "npy" else False

    # Crop threshold frame if a crop region is provided
    if len(crop) != 0:
        T_frame = T_frame[crop[0]:crop[1], crop[2]:crop[3], :]

    # Read binary NumPy data
    if bin_file:
        if len(correction) != 0:
            data = Bg_Correction(
                readBinary(Data_path, file_name),
                correction
            )
        else:
            data = emccd.readBinary(Data_path, file_name)

    # Read SIF data
    else:
        if len(correction) != 0:
            data = emccd.Bg_Correction(
                readSif(Data_path, file_name)[0],
                correction
            )
        else:
            data = emccd.readSif(Data_path, file_name)[0]

    # Crop photon-exposed data if required
    if len(crop) != 0:
        data = data[:, crop[0]:crop[1], crop[2]:crop[3]]

    # Convert corrected EMCCD frames into photon-counted frames
    cd = Thresholder(
        data,
        T_frame,
        max_photon_number=max_photon_number,
        dtype=dtype
    )

    print("Photon Counted!")

    # Free memory
    del data
    gc.collect()

    return cd

def JPD_4D(data: np.ndarray, shift: int = 0) -> np.ndarray:
    """
    Compute a four-dimensional joint probability distribution from photon-counted frames.

    For a frame stack `data[t, x, y]`, this computes the average outer product:

        JPD[x1, y1, x2, y2] = < data[t, x1, y1] data[t, x2, y2] >_t

    If `shift != 0`, the second frame is replaced by a shifted frame sequence. This
    is useful for estimating accidental/background coincidences.

    Parameters
    ----------
    data : np.ndarray
        Photon-counted frame stack with shape `(frames, rows, cols)`.
    shift : int, optional
        Temporal frame shift. If 0, same-frame correlations are computed. If nonzero,
        shifted-frame correlations are computed.

    Returns
    -------
    np.ndarray
        Four-dimensional JPD array with shape `(rows, cols, rows, cols)`.
    """
    data_gpu = cp.asarray(data)

    if shift == 0:
        jpd = cp.einsum("ijk,ipq->jkpq", data_gpu, data_gpu) / len(data_gpu)
    else:
        shifted_data = cp.roll(data_gpu, shift=shift, axis=0)
        jpd = cp.einsum("ijk,ipq->jkpq", data_gpu[1:], shifted_data[1:]) / (len(data_gpu) - 1)

    return cp.asnumpy(jpd)


def Hugo_JPD_4D(
    data: np.ndarray,
    segments: int = 20,
    max_photon: int = 1,
) -> np.ndarray:
    """
    Compute a background-subtracted, normalized 4D JPD in memory-safe segments.

    The function computes:

        jpd4d = JPD_same_frame - JPD_next_frame

    where `JPD_next_frame` is estimated using a one-frame temporal shift. The result
    is then normalized using:

        denom[x1,y1,x2,y2] = (max_photon - mean_frame[x1,y1])
                            (max_photon - mean_frame[x2,y2])

    and transformed as:

        log(1 + jpd4d / denom)

    This logarithmic transformation compresses the dynamic range of the recovered
    joint distribution.

    Parameters
    ----------
    data : np.ndarray
        Photon-counted frame stack with shape `(frames, rows, cols)`.
    segments : int, optional
        Number of segments used to divide the frame stack. Useful for reducing GPU
        memory load.
    max_photon : int, optional
        Maximum counted photon number used in the denominator normalization.

    Returns
    -------
    np.ndarray
        Background-subtracted and log-normalized 4D JPD.
    """
    rows, cols = data.shape[1], data.shape[2]

    jpdsf = np.zeros(shape=(rows, cols, rows, cols))
    jpdnf = np.zeros(shape=(rows, cols, rows, cols))

    for i in range(segments):
        if (i + 1) % 10 == 0:
            print("#", end="")

        start_idx = i * len(data) // segments
        end_idx = (i + 1) * len(data) // segments
        data_segment = data[start_idx:end_idx]

        jpdsf += JPD_4D(data_segment)
        jpdnf += JPD_4D(data_segment, shift=1)

    print()

    jpdsf /= segments
    jpdnf /= segments

    jpd4d = jpdsf - jpdnf

    del jpdsf
    del jpdnf
    gc.collect()

    mnc = cp.asarray(max_photon - np.mean(data, axis=0))
    denom = cp.asnumpy(cp.einsum("ij,lk->ijlk", mnc, mnc))

    with np.errstate(divide="ignore", invalid="ignore"):
        jpd4d = np.log(1 + jpd4d / denom)
        jpd4d = np.nan_to_num(jpd4d)

    return jpd4d


def JPD2D(
    jpd4d: np.ndarray,
    axis: Sequence[int] = (0, 2),
    show: bool = False,
) -> np.ndarray:
    """
    Reduce a 4D JPD into a 2D joint probability/correlation map.

    By default, this sums over axes `(0, 2)`, leaving a 2D map over the remaining
    two coordinates. This is useful for visualizing reduced spatial correlations.

    The diagonal is replaced by neighboring off-diagonal values to suppress the
    dominant self-correlation contribution.

    Parameters
    ----------
    jpd4d : np.ndarray
        Four-dimensional JPD array.
    axis : sequence of int, optional
        Axes over which to sum.
    show : bool, optional
        If True, display the resulting 2D map.

    Returns
    -------
    np.ndarray
        Reduced 2D JPD map.
    """
    jpd2d = np.sum(jpd4d, axis=tuple(axis))

    # Suppress diagonal/self-correlation line.
    for i in range(len(jpd2d)):
        jpd2d[i, i] = jpd2d[i - 1, i]

    if show:
        plt.imshow(jpd2d)
        plt.colorbar()
        plt.title("Reduced 2D JPD")
        plt.show()

    return jpd2d

