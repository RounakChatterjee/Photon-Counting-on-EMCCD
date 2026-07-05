# ============================================================
# EMCCD PHOTON COUNTING / THRESHOLDING MODEL
# ============================================================
#
# This code models the probability distribution of EMCCD output
# counts, including:
#
#   1. Readout noise
#   2. Clock-induced charge, CIC
#   3. Serial-register noise
#   4. EM gain multiplication
#   5. Photon statistics
#
# The main workflow is:
#
#   1. Load EMCCD frames using readBinary() or readSif()
#   2. Estimate the probability histogram using make_prob()
#   3. Fit the noise-only distribution using Noisefit()
#   4. Assign spatial beam probability using Beam(assign_prob=True)
#   5. Fit photon + noise distribution using Photonfit()
#   6. Calculate photon-count thresholds using Photon_Thresh()
#   7. Build pixel-wise threshold frames using create_T_frame()
#   8. Convert raw EMCCD frames into photon-counted data using Thresholder()
#
# ============================================================


import numpy as np
from numpy import convolve
import matplotlib.pyplot as plt

from scipy.stats import norm, erlang, poisson
from scipy.signal import fftconvolve as ftcv

from lmfit import Model

import torch
from torch.special import i1

from fft_conv_pytorch import fft_conv as tftcv

import sif_parser as pr
import json


# ============================================================
# GLOBAL PARAMETERS
# ============================================================

# Spatial probability distribution of the beam.
# This is assigned later using Beam(data, assign_prob=True).
bProb = np.array([])

# If True, fitting functions print parameters and diagnostic plots.
echo_flag = True

# Number of EMCCD multiplication registers.
# This is detector-specific.
Nregisters = 552

# Continuous count axis used internally for probability distributions.
# The EMCCD output probability distributions are calculated on this grid.
x_cont = np.arange(-5000, 5001, 1)


# ============================================================
# DEVICE SELECTION
# ============================================================

def get_device(echo_device=False):
    """
    Select the best available torch device.

    Priority:
        1. CUDA GPU
        2. Apple MPS GPU
        3. CPU

    Parameters
    ----------
    echo_device : bool
        If True, print the selected device.

    Returns
    -------
    torch.device
        Selected torch device.
    """

    if torch.cuda.is_available():
        dev = torch.device("cuda")

    elif torch.backends.mps.is_available():
        dev = torch.device("mps")

    else:
        dev = torch.device("cpu")

    if echo_device:
        print("Running on:", dev)

    return dev


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def readBinary(path, file, frames=0):
    """
    Read a binary numpy file containing EMCCD frames.

    Parameters
    ----------
    path : str
        Folder path.

    file : str
        File name.

    frames : int
        Number of frames to return.
        If frames=0, all frames are returned.

    Returns
    -------
    np.ndarray
        Data array with shape:

            frames x rows x columns
    """

    print("reading:" + path + file, end="\t")

    data = np.load(path + f"{file}").astype(np.int32)

    print("File Read!")

    if frames != 0:
        return data[:frames, :, :]
    else:
        return data


def readSif(path, file, frames=0):
    """
    Read Andor SIF file.

    Parameters
    ----------
    path : str
        Folder path.

    file : str
        SIF file name.

    frames : int
        Number of frames to return.
        If frames=0, all frames are returned.

    Returns
    -------
    data : np.ndarray
        EMCCD data array.

    info : dict or object
        Metadata returned by sif_parser.
    """

    print("reading:" + path + file, end="\t")

    data = pr.np_open(path + file)

    print("File Read!")

    if frames != 0:
        return data[0][:frames], data[1]
    else:
        return data[0], data[1]


# ============================================================
# HISTOGRAM / PROBABILITY ESTIMATION
# ============================================================

def make_prob(
    data,
    range=[400, 2500],
    show_plot=False,
    log_plot=True,
    make_offset=True
):
    """
    Convert EMCCD raw counts into a normalized probability histogram.

    The data is flattened and a histogram is calculated over the given
    count range. The histogram is normalized as a probability density.

    Parameters
    ----------
    data : np.ndarray
        EMCCD data array.

    range : list
        Count range over which the histogram is calculated.
        Example: [400, 2500]

    show_plot : bool
        If True, plot the histogram.

    log_plot : bool
        If True, plot the y-axis on a log scale.

    make_offset : bool
        If True, shift the x-axis so that the histogram maximum is at 0.

    Returns
    -------
    x : np.ndarray
        Count values, optionally offset-corrected.

    px : np.ndarray
        Normalized probability histogram.

    offset_x : float
        Count value corresponding to the histogram maximum.
    """

    hist = np.histogram(
        data.flatten(),
        range=range,
        bins=np.diff(range)[0],
        density=True
    )

    px = hist[0]
    x = hist[1][:-1]

    offset_x = x[np.argmax(px)]

    if make_offset:
        x -= offset_x

    if show_plot:
        plt.scatter(x, px, label=f"Offset = {offset_x}")
        plt.xlabel("Counts", size=15)
        plt.ylabel("Sample Probability", size=15)
        plt.legend(fontsize=15)

        if log_plot:
            plt.yscale("log")

        plt.show()

    return x, px, offset_x


# ============================================================
# PARAMETER LOADING
# ============================================================

def load_params(path, file):
    """
    Load fit parameters from a JSON file.

    NOTE
    ----
    Your original code had two functions named load_params().
    The second one overwrote the first. This version keeps only
    the path + file form.

    Parameters
    ----------
    path : str
        Folder path.

    file : str
        JSON file name.

    Returns
    -------
    np.ndarray
        Parameter values as float64 array.
    """

    with open(path + file, "r") as fp:
        params_dict = json.load(fp)

    return np.float64(np.array(list(params_dict.items()))[:, 1])


# ============================================================
# BASIC DISTRIBUTION MODELS
# ============================================================

def Erlang(x, n, G):
    """
    Erlang distribution used to model EM gain output for n electrons.

    For an EMCCD, after multiplication gain, an input electron does not
    produce a fixed output count. Instead, the gain process produces a
    stochastic output distribution, often modeled using an Erlang/gamma
    distribution.

    Parameters
    ----------
    x : np.ndarray
        Count axis.

    n : int
        Number of input electrons.

    G : float
        Mean EM gain.

    Returns
    -------
    np.ndarray
        Erlang probability density evaluated on x.
    """

    val = np.zeros(len(x))

    val = erlang.pdf(x, n, 0, G)

    # Manually force probability at x=0 to zero.
    # The zero-count probability is handled separately in the noise model.
    val[x == 0] = 0

    return val


# ============================================================
# NOISE MODEL
# ============================================================

def p_noise(x, sigma, pCIC, pser, pc):
    """
    EMCCD noise probability distribution.

    This model includes:

        1. Gaussian readout noise
        2. Clock-induced charge, CIC
        3. Serial-register noise

    The final noise distribution is obtained by convolution of these
    independent noise contributions.

    Parameters
    ----------
    x : np.ndarray
        Count axis where the final cropped probability distribution is required.

    sigma : float
        Standard deviation of Gaussian readout noise.

    pCIC : float
        Probability of clock-induced charge.

    pser : float
        Probability of serial-register noise per register.

    pc : float
        Per-register multiplication probability.

    Returns
    -------
    np.ndarray
        Normalized noise probability distribution evaluated on x.
    """

    if echo_flag:
        print(sigma, pCIC, pser, pc)

    # Total EM gain after Nregisters multiplication stages.
    G = (1 + pc) ** Nregisters

    # --------------------------------------------------------
    # 1. Gaussian readout noise
    # --------------------------------------------------------
    c1 = norm.pdf(x_cont, 0, sigma)

    # --------------------------------------------------------
    # 2. Clock-induced charge, CIC
    # --------------------------------------------------------
    c2 = pCIC * Erlang(x_cont, 1, G)

    # Probability of no CIC event.
    c2[x_cont == 0] = 1 - pCIC

    # --------------------------------------------------------
    # 3. Serial-register noise
    # --------------------------------------------------------
    c3 = np.zeros(len(x_cont))

    # Serial noise can be introduced at different multiplication stages.
    # A noise electron generated later experiences a smaller effective gain.
    for k in range(1, Nregisters + 1):

        effective_gain = (1 + pc) ** (Nregisters - k)

        c3 += pser * Erlang(x_cont, 1, effective_gain)

    # Probability that no serial-register noise event occurs.
    c3[x_cont == 0] = 1.0 - Nregisters * pser

    # --------------------------------------------------------
    # Total noise distribution
    # --------------------------------------------------------
    val = convolve(
        c1,
        convolve(c2, c3, mode="same"),
        mode="same"
    )

    # Crop the global x_cont distribution to the requested x range.
    indices = np.where((x_cont >= x[0]) & (x_cont <= x[-1]))

    # Normalize on the cropped region.
    return val[indices] / np.sum(val[indices])


# ============================================================
# PHOTON MODEL
# ============================================================

def p_photon(x, pc, mu):
    """
    EMCCD photon probability distribution.

    This calculates the output count probability distribution due to
    photons arriving at the EMCCD.

    The number of detected photons follows Poisson statistics with mean mu.
    After EM multiplication, the output has a Bessel-function form.

    Parameters
    ----------
    x : np.ndarray
        Count axis.

    pc : float
        Per-register multiplication probability.

    mu : np.ndarray
        Mean photon number values.

    Returns
    -------
    np.ndarray
        Photon probability distribution.

        If len(mu) != 1:
            returns the average probability over all mu values.

        If len(mu) == 1:
            returns the probability distribution for that single mu.
    """

    device = get_device()

    # Only positive count values enter the continuous EM-gain expression.
    x_t = torch.tensor(x[x > 0], device=device, dtype=torch.float32)

    mu_t = torch.tensor(mu, device=device, dtype=torch.float32)

    # Create mesh for evaluating probability for many mu values.
    X, Mu = torch.meshgrid(x_t, mu_t, indexing="xy")

    # Total EM gain.
    G = (1 + pc) ** Nregisters

    # Bessel-function form of amplified photon-count distribution.
    val = torch.exp(-Mu - X / G) * torch.sqrt(Mu / (G * X)) * i1(
        2 * torch.sqrt((Mu * X / G))
    )

    # Add discrete zero-count probability exp(-mu).
    val = torch.column_stack((torch.exp(-mu_t), val))

    # Pad negative-x region with zero probability.
    val = torch.nn.functional.pad(
        val,
        (len(x[x < 0]), 0),
        mode="constant",
        value=0.0
    )

    # Remove possible NaNs from numerical issues.
    val = torch.nan_to_num(val)

    if len(mu) != 1:
        # Average over all spatial pixels / number of pixels.
        val = torch.mean(val, axis=0).detach().cpu().numpy()
        return val

    else:
        return val.detach().cpu().numpy()[0]


# ============================================================
# COMBINED PHOTON + NOISE MODEL
# ============================================================

def p_combined(x, mu, sigma, pCIC, pser, pc):
    """
    Full EMCCD output probability model.

    This combines:

        1. EMCCD noise distribution
        2. Photon-induced distribution

    The total probability is calculated by convolution:

        p_total = p_noise * p_photon

    The spatial beam probability bProb is used to generate a list of
    pixel-dependent photon means:

        mu_pixel = mu * bProb_pixel

    Parameters
    ----------
    x : np.ndarray
        Count axis for the final probability distribution.

    mu : float
        Total mean photon number scale.

    sigma : float
        Readout noise standard deviation.

    pCIC : float
        Clock-induced charge probability.

    pser : float
        Serial-register noise probability.

    pc : float
        Per-register multiplication probability.

    Returns
    -------
    np.ndarray
        Normalized photon + noise probability distribution.
    """

    # Pixel-wise mean photon numbers.
    # Requires bProb to have already been assigned.
    mu_list = (mu * bProb).flatten()

    # Noise distribution calculated on x_cont.
    noise_vals = p_noise(x_cont, sigma, pCIC, pser, pc)

    # Photon distribution averaged over the beam probability.
    photon_vals = p_photon(x_cont, pc, mu_list)

    # Convolve noise and photon distributions.
    val = ftcv(noise_vals, photon_vals, mode="same")

    # Crop back to desired x range.
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
            np.sum(val)
        )

        plt.plot(x, return_val)
        plt.yscale("log")
        plt.show()

    return return_val


# ============================================================
# BEAM PROFILE
# ============================================================

def Beam(data, assign_prob=False):
    """
    Show beam profile and optionally assign spatial beam probability bProb.

    If data is a 3D frame stack, the average over frames is used.

    Parameters
    ----------
    data : np.ndarray
        Either a 2D image or a 3D frame stack.

    assign_prob : bool
        If True, normalize the beam profile and assign it to global bProb.

    Returns
    -------
    None
    """

    if len(data.shape) == 3:
        beam = np.mean(data, axis=0)
    else:
        beam = data

    if assign_prob:
        beam[beam < 0] = 0

        P = beam / np.sum(beam)

        global bProb
        bProb = P

        plt.imshow(bProb)
        plt.title("Assigned Beam Probability")
        plt.colorbar()
        plt.show()

    else:
        plt.imshow(beam)
        plt.title("Beam Profile")
        plt.colorbar()
        plt.show()


# ============================================================
# BACKGROUND CORRECTION
# ============================================================

def Bg_Correction(data, correction=[]):
    """
    Correct row/column background structure in EMCCD frames.

    If no correction is provided, the correction image is estimated as:

        Correction = Row contribution + Column contribution - Mean image value

    Parameters
    ----------
    data : np.ndarray
        EMCCD frame stack with shape:

            frames x rows x columns

    correction : np.ndarray or list
        Precomputed correction image.
        If empty, a correction image is estimated from the data.

    Returns
    -------
    corrected_data : np.ndarray
        Background-corrected data.

    Correction : np.ndarray
        Returned only when correction is calculated internally.
    """

    corrected_data = np.zeros(data.shape, dtype=np.float32)

    if len(correction) == 0:

        row_vals = np.mean(data, axis=(0, 2))
        col_vals = np.mean(data, axis=(0, 1))
        img_mean = np.mean(data)

        R, C = np.meshgrid(col_vals, row_vals)

        B = img_mean * np.ones(shape=(len(row_vals), len(col_vals)))

        Correction = R + C - B

        corrected_data[:] = data[:] - Correction

        return corrected_data, Correction

    else:
        corrected_data[:] = data[:] - correction

        return corrected_data


# ============================================================
# FITTING WEIGHTS
# ============================================================

def weight(prob: np.ndarray, precision: int = 5) -> np.ndarray:
    """
    Generate fitting weights based on probability order of magnitude.

    This is useful for fitting probability distributions on a log scale.
    Without weighting, least-squares fitting is dominated by the peak of
    the distribution and ignores the low-probability tail.

    Parameters
    ----------
    prob : np.ndarray
        Probability values.

    precision : int
        Maximum order-of-magnitude weighting.
        Values below 10^-precision receive weight 10^precision.

    Returns
    -------
    weights : np.ndarray
        Weight array of same length as prob.
    """

    weights = np.ones(len(prob))

    for i in range(precision):

        mask = (prob > 10 ** (-i - 1)) & (prob <= 10 ** (-i))

        weights[mask] = (10 ** i) * weights[mask]

    weights[prob <= 10 ** (-precision)] = 10 ** precision

    return weights


# ============================================================
# NOISE FITTING
# ============================================================

def Noisefit(
    px,
    x,
    weights,
    params,
    vary_flags={
        "sigma": True,
        "pCIC": True,
        "pser": True,
        "pc": True
    },
    method="leastsq"
):
    """
    Fit the noise-only EMCCD probability distribution.

    Parameters
    ----------
    px : np.ndarray
        Measured probability histogram.

    x : np.ndarray
        Count axis corresponding to px.

    weights : np.ndarray
        Fitting weights.

    params : array-like
        Initial parameter guesses:

            [sigma, pCIC, pser, pc]

    vary_flags : dict
        Flags specifying which parameters are allowed to vary.

    method : str
        lmfit minimization method.

    Returns
    -------
    lmfit.model.ModelResult
        Fit result object.
    """

    fit_model = Model(p_noise, nan_policy="omit")

    sigma, pCIC, pser, pc = params

    fit_model.set_param_hint(
        "sigma",
        value=sigma,
        min=0,
        max=50.0,
        vary=vary_flags["sigma"]
    )

    fit_model.set_param_hint(
        "pCIC",
        value=pCIC,
        min=0,
        max=1,
        vary=vary_flags["pCIC"]
    )

    fit_model.set_param_hint(
        "pser",
        value=pser,
        min=0,
        max=1,
        vary=vary_flags["pser"]
    )

    fit_model.set_param_hint(
        "pc",
        value=pc,
        min=0,
        max=1,
        vary=vary_flags["pc"]
    )

    result = fit_model.fit(
        px,
        x=x,
        weights=weights,
        method=method
    )

    return result


# ============================================================
# PHOTON + NOISE FITTING
# ============================================================

def Photonfit(
    px,
    x,
    weights,
    params,
    vary_flags={
        "mu": True,
        "sigma": True,
        "pCIC": True,
        "pser": True,
        "pc": True
    },
    method="leastsq"
):
    """
    Fit the full photon + noise EMCCD probability distribution.

    Parameters
    ----------
    px : np.ndarray
        Measured probability histogram.

    x : np.ndarray
        Count axis.

    weights : np.ndarray
        Fitting weights.

    params : array-like
        Initial parameter guesses:

            [mu, sigma, pCIC, pser, pc]

    vary_flags : dict
        Flags specifying which parameters are allowed to vary.

    method : str
        lmfit minimization method.

    Returns
    -------
    lmfit.model.ModelResult
        Fit result object.
    """

    fit_model = Model(p_combined, nan_policy="omit")

    mu, sigma, pCIC, pser, pc = params

    fit_model.set_param_hint(
        "mu",
        value=mu,
        min=0,
        max=1e6,
        vary=vary_flags["mu"]
    )

    fit_model.set_param_hint(
        "sigma",
        value=sigma,
        min=0,
        max=30.0,
        vary=vary_flags["sigma"]
    )

    fit_model.set_param_hint(
        "pCIC",
        value=pCIC,
        min=0,
        max=1,
        vary=vary_flags["pCIC"]
    )

    fit_model.set_param_hint(
        "pser",
        value=pser,
        min=0,
        max=1,
        vary=vary_flags["pser"]
    )

    fit_model.set_param_hint(
        "pc",
        value=pc,
        min=0,
        max=1,
        vary=vary_flags["pc"]
    )

    result = fit_model.fit(
        px,
        x=x,
        weights=weights,
        method=method
    )

    return result


# ============================================================
# PLOTTING FIT RESULTS
# ============================================================

def plot_fit(
    x,
    result,
    prob,
    log_plot=True,
    xlim=[],
    ylim=[],
    save_options={"fig": False, "parameters": False},
    save_file_name=""
):
    """
    Plot measured probability and lmfit best-fit result.

    Parameters
    ----------
    x : np.ndarray
        Count axis.

    result : lmfit.model.ModelResult
        Fit result returned by Noisefit() or Photonfit().

    prob : np.ndarray
        Measured probability distribution.

    log_plot : bool
        If True, plot probability on log scale.

    xlim : list
        Optional x-axis limit.

    ylim : list
        Optional y-axis limit.

    save_options : dict
        Dictionary with keys:

            "fig" : bool
            "parameters" : bool

    save_file_name : str
        Base file name for saving plot and parameters.

    Returns
    -------
    dict
        Best-fit parameter values.
    """

    plt.scatter(
        x,
        prob,
        color="red",
        marker="x",
        label="EMCCD Data"
    )

    plt.plot(
        x,
        result.best_fit,
        lw=3,
        label="Fit",
        color="#00FF00"
    )

    plt.xlabel("Counts", size=15)
    plt.ylabel("Probability", size=15)
    plt.legend(fontsize=15)

    if log_plot:
        plt.yscale("log")

    if len(xlim) != 0:
        plt.xlim(xlim)

    if len(ylim) != 0:
        plt.ylim(ylim)

    if save_options["fig"]:
        plt.savefig(save_file_name + ".png", dpi=1600)

    plt.show()

    params = result.best_values

    print("Data max = ", x[np.argmax(prob)])
    print("Fit max = ", x[np.argmax(result.best_fit)])

    print(f"{result.fit_report()}")

    if save_options["parameters"]:
        with open(save_file_name + "_params.json", "w") as fp:
            json.dump(result.best_values, fp)

    return params


def plot_fit_data(
    x,
    prob,
    params,
    log_plot=True,
    xlim=[],
    ylim=[]
):
    """
    Plot measured data against model calculated from explicit parameters.

    Unlike plot_fit(), this function does not require an lmfit result object.

    Parameters
    ----------
    x : np.ndarray
        Count axis.

    prob : np.ndarray
        Measured probability.

    params : array-like
        Parameters for p_combined():

            [mu, sigma, pCIC, pser, pc]

    log_plot : bool
        If True, plot probability on log scale.

    xlim : list
        Optional x-axis limit.

    ylim : list
        Optional y-axis limit.

    Returns
    -------
    None
    """

    plt.scatter(
        x,
        prob,
        color="red",
        marker="x",
        label="EMCCD Data"
    )

    fit_x = p_combined(x, *params)

    plt.plot(
        x,
        fit_x,
        lw=3,
        label="Fit",
        color="#00FF00"
    )

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


# ============================================================
# PHOTON THRESHOLD CALCULATION
# ============================================================

def Photon_Thresh(N, params):
    """
    Calculate threshold count values separating photon-number states.

    The threshold between n-1 and n photons is found from the crossing
    point of the corresponding posterior probabilities.

    Parameters
    ----------
    N : int
        Maximum photon number up to which thresholds are calculated.

    params : array-like
        Full parameter list:

            [mu, sigma, pCIC, pser, pc]

    Returns
    -------
    val : np.ndarray
        Threshold values.

        val[n-1] gives the threshold separating n-1 and n photons.
    """

    mu = params[0]

    gain = (1 + params[-1]) ** Nregisters

    # Noise probability.
    pNx = p_noise(x_cont, *params[1:])

    # Photon probability for mean photon number mu.
    px = p_photon(
        x=x_cont,
        pc=params[-1],
        mu=np.array([mu])
    )

    # Total noise + photon probability.
    pxx = np.convolve(pNx, px, mode="same")

    val = np.zeros(N)

    p_prev = np.zeros(len(x_cont))

    for n in range(N + 1):

        if n == 0:

            # Prior probability of zero photons.
            p0mu = poisson.pmf(n, mu)

            # Posterior distribution for zero photons.
            p0x = (p0mu * pNx) / pxx

            p0x[np.isnan(p0x)] = 0

            p_prev = p0x

        else:

            # EMCCD output distribution for exactly n input photons.
            pxn = Erlang(x_cont, n, gain)

            # Poisson probability of n photons.
            pnmu = poisson.pmf(n, mu)

            # Posterior distribution for n photons.
            pnx = np.convolve(pNx, pxn * pnmu, mode="same") / pxx

            pnx[np.isnan(pnx)] = 0

            # Threshold is first crossing where p_n exceeds p_{n-1}.
            crossing_indices = np.where((pnx - p_prev) > 0)[0]

            if len(crossing_indices) > 0:
                val[n - 1] = x_cont[crossing_indices[0]]
            else:
                val[n - 1] = np.nan

            p_prev = pnx

    return val


# ============================================================
# APPLY THRESHOLDS TO DATA
# ============================================================

def Thresholder(
    data,
    T_frame,
    max_photon_number=1,
    dtype=np.int32
):
    """
    Convert raw EMCCD counts into photon-counted data.

    For each pixel and frame, the output count is incremented whenever
    the raw EMCCD value crosses a threshold.

    Example:
        If max_photon_number = 3 and the pixel crosses two thresholds,
        the output photon count becomes 2.

    Parameters
    ----------
    data : np.ndarray
        Raw EMCCD data with shape:

            frames x rows x columns

    T_frame : np.ndarray
        Pixel-wise threshold frame with shape:

            rows x columns x number_of_thresholds

    max_photon_number : int
        Maximum photon number to count.

    dtype : data type
        Output data type.

    Returns
    -------
    Counted_data : np.ndarray
        Photon-counted data with same shape as input data.
    """

    available_thresholds = len(T_frame[0, 0, :])

    if max_photon_number > available_thresholds:
        print(
            "Not enough Threshold provided, reverting to maximum available = ",
            available_thresholds
        )

        max_photon_number = available_thresholds

    Counted_data = np.zeros(data.shape, dtype=dtype)

    for n in range(max_photon_number):

        Counted_data[:, :, :] += dtype(data[:, :, :] >= T_frame[:, :, n])

    return Counted_data


# ============================================================
# CREATE PIXEL-WISE THRESHOLD FRAME
# ============================================================

def create_T_frame(m_vals, params, frame_size=[200, 200]):
    """
    Create a pixel-wise threshold matrix.

    Each pixel has a different mean photon number:

        mu_pixel = mu_total * bProb_pixel

    Therefore, each pixel can have different photon-count thresholds.

    This function precomputes thresholds for a list of possible mean photon
    values m_vals, then assigns the nearest threshold set to each pixel.

    Parameters
    ----------
    m_vals : np.ndarray
        Sorted array of mean photon values for which thresholds are precomputed.

    params : array-like
        Full parameter list:

            [mu, sigma, pCIC, pser, pc]

    frame_size : list
        Frame shape:

            [rows, columns]

    Returns
    -------
    T_frame : np.ndarray
        Pixel-wise threshold matrix with shape:

            rows x columns x number_of_thresholds
    """

    print(
        f"Mean Photon per frame per pixel: {params[0] * np.mean(bProb)}"
        f"\n params = {params[1:]}"
    )

    # pixel_params has same structure as params:
    # [mu_pixel, sigma, pCIC, pser, pc]
    pixel_params = np.append(0, params[1:])

    print(len(m_vals))

    T_vals = []

    for m in m_vals:

        pixel_params[0] = m

        # Choose maximum photon number for which thresholds are needed.
        N_thresh = int(m + 1.5 * np.ceil(np.sqrt(m)))

        T = Photon_Thresh(N_thresh, pixel_params)

        T_vals.append(T)

    # Store all threshold arrays in a rectangular matrix.
    # Shorter threshold arrays are padded with a large number.
    max_len = len(T_vals[-1])

    T_mat = np.zeros(shape=(len(T_vals), max_len))

    for i in range(len(T_vals)):

        T_mat[i] = np.append(
            T_vals[i],
            5000 * np.ones(max_len - len(T_vals[i]))
        )

    # Pixel-wise photon mean values.
    mu_vals = params[0] * bProb.flatten()

    # Find nearest/precomputed threshold index.
    indices = np.searchsorted(m_vals, mu_vals)

    # Important safety correction:
    # searchsorted can return len(m_vals), which would be out of range.
    indices = np.clip(indices, 0, len(m_vals) - 1)

    T_frame = T_mat[indices].reshape(
        frame_size[0],
        frame_size[1],
        max_len
    )

    return T_frame