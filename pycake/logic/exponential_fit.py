import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit

def failed_dict(additional_entries={}):
    fd = {'tau_m': np.nan, 'V_rest': np.nan, 'height': np.nan, 'std_tau': np.nan,
        'reduced_chisquare': np.nan, 'pcov': np.nan, 'ier': 0}
    fd.update(additional_entries)
    return fd


def exp_func(t, tau, height, V_rest):
    return height * np.exp(-(t - t[0]) / tau) + V_rest


def get_initial_parameters(t, v, V_rest=None):
    if V_rest is None:
        V_rest = v[-1] # Take last point of trace if no V_rest is specified
    height = v[0] - V_rest
    try:
        # get first element where trace is below 1/e of initial value
        # This does not always work
        tau = t[v-V_rest < ((v[0]-V_rest) / np.exp(1))][0]
    except IndexError:  # If trace never goes below 1/e, use some high value
        tau = 3 * t[-1]
    return [tau, height, V_rest]


def fit_exponential(t, v, std=None, n_chunks=None, full_output=False, V_rest_init=None):
    """ Fit an exponential function to a trace. 
        Args:
            t: time axis
            v: voltage trace
            std: standard deviation of the voltage trace
                 this is used to approximate the error of tau
            n_chunks: if the trace was an average over many repetitions, specify HOW many.
            full_output: returns the infodict. This can be as large as the trace itself!
            V_rest_init: If not specified, initial V_rest will be last point in trace"""
    x0 = get_initial_parameters(t, v)

    # If std of trace and number of reps is given, calculate std of averaged trace
    if not None in [std, n_chunks]:
        std = std / np.sqrt(n_chunks)

    try:
        expf, pcov, infodict, errmsg, ier = curve_fit(
            exp_func,
            t,
            v,
            x0,
            sigma=std,
            full_output=True)
    except (ValueError, ZeroDivisionError, FloatingPointError) as e:
        return failed_dict(e, {"V_rest": x0[2]})

    tau = expf[0]
    height = expf[1]
    V_rest = expf[2]

    DOF = len(t) - len(expf)
    red_chisquare = sum(infodict["fvec"] ** 2) / (DOF)

    # Sanity checks:
    if not isinstance(pcov, np.ndarray):
        e = 'pcov not array'
        return failed_dict({'error': e, "V_rest": x0[2], 'infodict': infodict})
    if isinstance(pcov, np.ndarray) and pcov[0][0] < 0:
        e = 'pcov negative'
        return failed_dict({'error': e, "V_rest": x0[2], 'infodict': infodict})
    if (ier < 1) or (tau > 1) or (tau < 0):
        e = 'ier < 1 or tau out of bounds'
        return failed_dict({"error": e, "V_rest": x0[2], "infodict": infodict})

    tau_std = np.sqrt(pcov[0][0])

    if not full_output:
        infodict = None

    return {'tau_m': tau, 'V_rest': V_rest, 'height': height, 'std_tau': tau_std,
            'reduced_chisquare': red_chisquare, 'pcov': pcov, 'infodict': infodict, 'ier': ier}


def get_decay_fit_range(trace, adc_freq, fwidth=64):
    """Detects decaying part of trace (for exponential fit).
       Returns subtrace (v, t) in this range.

    Args:
        trace: the averaged trace
        adc_freq: frequency of the adc
        fwidth: filter width for smoothing

    Returns: decaying trace and corresponding times
        cut_t, cut_v"""

    dt = 1./adc_freq
    vsmooth = ndimage.gaussian_filter(trace, fwidth)
    # Sobel filter finds edges
    edges = ndimage.sobel(vsmooth, axis=-1, mode='nearest')
    fitstart, fitstop = np.argmin(edges), np.argmax(edges)

    fitstop = fitstop - fitstart

    if fitstop < 0:
        fitstop += len(trace)

    trace = np.roll(trace, -fitstart)

    fitstart = 0
    trace_cut = trace[fitstart:fitstop]
    fittimes = np.arange(fitstart, fitstop)*dt

    return fittimes[:-80], trace_cut[:-80]  # sobel tends to make trace too long --> cutoff
