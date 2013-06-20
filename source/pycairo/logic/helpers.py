"""Helper functions"""

import pycalibtic
from scipy import optimize
import numpy as np


def create_pycalibtic_polynomial(coefficients):
    """Create a pycalibtic.Polynomial from a list of coefficients.

    Order: [c0, c1, c2, ...] resulting in c0*x^0 + c1*x^1 + c2*x^2 + ..."""
    data = pycalibtic.vector_less__double__greater_()
    for i in coefficients:
        data.append(i)
    return pycalibtic.Polynomial(data)


class Helpers(object):
    """Helper functions used in interfaces/hardware.py"""
########### Helpers functions ##############

    ## Fit PSP
    # @param tm Time array
    # @param psp Voltage array
    # @param parameter Parameter to be fitted, can be 'tausynx' or 'tausyni'
    # @param Esyn Synaptic reversal potential
    # @param Cm Capacitance of the neuron
    def fit_psp(self, tm, psp, parameter, Esyn, Cm):
        # Seperate between exc and inh
        if (parameter == 'tausynx'):

            # Measure base
            base = min(psp)

            # Initial fit params (see pfit below)
            p0 = [2, 1, base, 2]

            # Calc error function
            errfunc = lambda p, t, y: self.psp_trace_exc(p, t) - y

            pfit, success = optimize.leastsq(errfunc, p0[:], args=(tm, psp))
            tau_syn, tau_eff, v_offset, t_offset = pfit

            trace_fit = self.psp_trace_exc(pfit, tm)

        if (parameter == 'tausyni'):

            # Measure base
            base = max(psp)

            # Initial fit params (see pfit below)
            p0 = [2, 1, base, 2]

            # Calc error function
            errfunc = lambda p, t, y: self.psp_trace_inh(p, t) - y

            pfit, success = optimize.leastsq(errfunc, p0[:], args=(tm, psp))
            tau_syn, tau_eff, v_offset, t_offset = pfit

            trace_fit = self.psp_trace_inh(pfit, tm)

        # Weight calculation
        e_rev_E = Esyn
        # nF
        Cm = Cm
        v_rest = base
        calc_weights = 40*Cm*(1./tau_eff - 1./tau_syn)/(e_rev_E - v_rest)

        return pfit[0]

    ## Fit PSP function for exc PSPs
    # @param p Array with the parameters of the fit
    # @param t Time array
    def psp_trace_exc(self, p, t):
        return self.theta(t - p[3])*40*(np.exp(-(t-p[3])/p[0]) - np.exp(-(t-p[3])/p[1])) + p[2]

    ## Fit PSP function for inh PSPs
    # @param p Array with the parameters of the fit
    # @param t Time array
    def psp_trace_inh(self, p, t):
        return self.theta(t - p[3])*-40*(np.exp(-(t-p[3])/p[0]) - np.exp(-(t-p[3])/p[1])) + p[2]

    ## Fit PSP
    # @param t Time array
    def theta(self, t):
        return (t > 0) * 1.

    ## Fit tw trace
    # @param p Array with the parameters of the fit
    # @param t Time array
    def fit_tw_trace(self, p, t):
        return (p[2]*np.cos(p[1]*t)+p[3]*np.sin(p[1]*t))*np.exp(-(p[0]*t)) + p[4]

    ## Fit tw trace
    # @param tm Time array
    # @param trace Voltage array
    # @param C Capacitance of the neuron
    # @param gL Membrane leakage conductance of the neuron
    def tw_fit(self, tm, trace, C, gL):
        # Initial fit params (see pfit below)
        # l, w, A, B, v_inf, dt
        p0 = [300000, 400000, -0.02, 0.15, 0.63]

        # Fit the data
        errfunc = lambda p, t, y: self.fit_tw_trace(p, t) - y
        tfit, success = optimize.leastsq(errfunc, p0[:], args=(tm, trace))
        l, w, A, B, v_inf = tfit

        trace_fit = self.fit_tw_trace(tfit, tm)

        # Calc tw
        tm = C*1e-12/(gL*1e-9)
        tw = tm/(2*tfit[0]*tm-1)

        return tw*1e6

    ## Fit trace to find dT
    # @param p Array of parameter for the fit function
    # @param t Time array
    def fit_exp_trace(self, p, t):
        return p[0]*np.exp(t/p[1])

    ## Fit trace to find dT
    # @param voltage Voltage array
    # @param current Current array
    def exp_fit(self, voltage, current):
        # Initial fit params (see pfit below)
        # l, w, A, B, v_inf, dt
        p0 = [0.2e-10, 12e-3]

        # Fit the data
        errfunc = lambda p, t, y: self.fit_exp_trace(p, t) - y
        tfit, success = optimize.leastsq(errfunc, p0[:], args=(voltage, current))

        i0, dT = tfit

        trace_fit = self.fit_exp_trace(tfit, voltage)

        return dT*1e3

    ## Calc dT from voltage trace
    # @param t Time array
    # @param v Voltage array
    # @param C Capacitance of the neuron
    # @param gL Membrane leakage conductance of the neuron
    # @param EL Membrane resting potential of the neuron
    def calc_dT(self, t, v, C, gL, EL):
        # Calculate exp current
        diff = []
        for i in range(1, len(v)):
            diff.append((v[i]-v[i-1])/(t[i]-t[i-1]))

        exp = []
        for i in range(len(diff)):
            exp.append(C*1e-12/(gL*1e-9)*diff[i] + (v[i]-EL*1e-3))

        # Cut the right part
        end_found = False
        end = 0
        for i in range(1, len(exp)):
            if (((exp[i] - exp[i-1]) > 2) and end_found is False):
                end_found = True
                end = i-10

        v = v[:end]
        exp = exp[:end]
        t = t[:end]

        # Cut the left part
        new_exp = []
        new_v = []
        new_t = []

        for i in range(len(exp)):
            if (exp[i] > 0.015):
                new_exp.append(exp[i])
                new_v.append(v[i])
                new_t.append(t[i])

        v = new_v
        exp = new_exp
        t = new_t

        return self.exp_fit(v, exp)


def marco2halbe(parameters):
    """Converts Marco's parameter names to HALbe parameter names."""

    return {
        "E_l": parameters["EL"],
        "E_syni": parameters["Esyni"],
        "E_synx": parameters["Esynx"],
        "I_bexp": parameters["Ibexp"],
        "I_convi": parameters["gsyni"],
        "I_convx": parameters["gsynx"],
        "I_fire": parameters["b"],
        "I_gl": parameters["gL"],
        "I_gladapt": parameters["a"],
        "I_intbbi": parameters["Iintbbi"],
        "I_intbbx": parameters["Iintbbx"],
        "I_pl": parameters["tauref"],
        "I_radapt": parameters["tw"],
        "I_rexp": parameters["dT"],
        "I_spikeamp": parameters["Ispikeamp"],
        "V_exp": parameters["Vexp"],
        "V_syni": parameters["Vsyni"],
        "V_synx": parameters["Vsynx"],
        "V_syntci": parameters["tausyni"],
        "V_syntcx": parameters["tausynx"],
        "V_t": parameters["Vt"]
    }
