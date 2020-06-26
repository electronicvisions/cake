import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from numba import jit


@jit(nopython=True)
def gs(t, w, tau_e):
    """function that calculates the conducance to the reversal potential

    Args:
        t: time after spike
        w: synaptic weight
        tau_e: synaptic time constant

    Returns:
        the value of the synaptic conductance at the given time
    """
    return w * np.exp(-t / tau_e)


@jit(nopython=True)
def dudt(u, t, g_l, v_l, w, tau_e, v_rev):
    """ calculated the differential voltage in respect to time i.e. du/dt

    Args:
        u: voltage at time t
        t: time
        g_l: leakage conductance
        v_l: leak potential
        w: weight
        tau_e: synaptic time constant
        v_rev: reversal potential

    Returns:
        du/dt for voltage u and time t at given parameters
    """
    dudt = -g_l * (u - v_l) - gs(t, w, tau_e) * (u - v_rev)
    return dudt


def fit_func(t, v_l, v_rev, tau_l, tau_e, w):
    """ numerically solves the COBA differential equation for given time and parameter

    Args:
        t: array of times
        v_l: leak potential
        v_rev: reversal potenial
        tau_l: membrane time constant
        tau_e: synaptic time constant
        w: weight

    Returns:
        the solution to the differential equation

    calculates the leakage conductance by g_l =  1 / tau_l as the capacity is set to 1
    expects the voltage to be v_l at time t = 0
    """
    y0 = v_l
    g_l = 1 / tau_l
    v_ode = np.concatenate(odeint(dudt, y0, t, mxstep=5000000, args=(g_l, v_l, w, tau_e, v_rev)))
    return v_ode


@jit(nopython=True)
def w_approx(h, tau_syn, tau_m, v_rev, v_l, cm):
    """function that gives an approximate value for the synaptic weight,
    uses a approximation based on the high conductance state and a
    artificial saturation

    Args:
        h: height of PSP to be fitted
        tau_syn: aproximate synaptic time constant
        tau_m: approximate mebrane time constant
        v_rev: approximate reversal potential
        v_l: approximate resting potential
        cm: membrane capacity

    Returns:
        A approximation for the synaptic weight to be used for start parameters in the fit

    """
    tau = tau_syn / tau_m
    tau_g = (1 / tau_syn - 1 / tau_m)**(-1.)
    h_max = v_rev - v_l
    kappa = tau_g / cm * (tau**(tau / (1 - tau)) - tau ** (1 / (1 - tau)))
    return np.log(1 - h / h_max) / -kappa


def fit(time, v, p0, bounds):
    """fits the COBA solution to a voltage course
    Args:
        time: array of the time
        v: array of the voltage

        p_0: array of the fitting startparameters (length of 5:
        [v_l, v_rev, tau_l, tau_e, w] with repective start parameters)

        bounds: array of the bounds of the fitted parameters first the
        lower bounds then the upper bounds ((v_l_low, v_rev_low,
        tau_l_low, tau_e_low, w_low),(v_l_high, v_rev_high, ...))
        needs +-np.inf for no bounds

        voltage needs to start at beginning of PSP, time needs to
        start at t = 0, time cannot be in too small steps, otherwise
        fit will not work.

    Returns:
        array of an array with the fitted parameters and the covariance matrix of the fit

    """
    popt, pcov = curve_fit(fit_func, time, v, p0,
                           maxfev=5000, bounds=bounds, check_finite=False)

    return popt, pcov
