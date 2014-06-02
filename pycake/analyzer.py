import numpy as np
import pylab
import pylogging
import scipy.signal
from scipy.optimize import curve_fit
from pycake.helpers.TraceAverager import createTraceAverager
from pycake.helpers.peakdetect import peakdet

# Import everything needed for saving:


class Analyzer(object):
    """ Takes a measurement and analyses it.
    """
    logger = pylogging.get("pycake.analyzer")

    def __call__(self, t, v, neuron):
        """ Returns a dictionary of results:
            {neuron: value}
        """
        raise NotImplemented

class MeanOfTraceAnalyzer(Analyzer):
    """ Analyzes traces for E_l measurement.
    """
    def __call__(self, t, v, neuron):
        return { "mean" : np.mean(v),
                 "std"  : np.std(v),
                 "max"  : np.max(v),
                 "min"  : np.min(v)}

class V_reset_Analyzer(Analyzer):
    """ 
    """
    def __call__(self, t,v, neuron):
        baseline, delta_t = self.find_baseline(t,v)
        
        return { "baseline" : baseline,
                 "delta_t"  : delta_t}

    def find_baseline(self, t,v):
            """ find baseline of trace

                t - list of time stamps
                v - corresponding values

                returns baseline and time between values in unit index (do t[delta_t] to get delta in units of time)
            """

            std_v = np.std(v)

            # to be tuned
            drop_threshold = -std_v/2
            min_drop_distance = 5
            start_div = 10
            end_div = 5

            # find right edge of spike --------------------------------------------
            diff = np.diff(v)
            drop = np.where(diff < drop_threshold)[0]
            #----------------------------------------------------------------------

            # the differences of the drop position n yields the time between spikes
            drop_diff = np.diff(drop)
            # filter consecutive values
            drop_diff_filtered = drop_diff[drop_diff > min_drop_distance]
            # take mean of differences
            delta_t = np.mean(drop_diff_filtered)
            #----------------------------------------------------------------------

            # collect baseline voltages -------------------------------------------
            only_base = []
            last_n = -1

            baseline = 0
            N = 0

            for n in drop[np.where(np.diff(drop) > min_drop_distance)[0]]:

                # start some time after spike and stop early to avoid taking rising edge
                start = n+int(delta_t/start_div)
                end = n+int(delta_t/end_div)

                if start < len(v) and end < len(v):
                    baseline += np.mean(v[start:end])
                    N += 1

            # take mean
            if N:
                baseline /= N
            else:
                baseline = np.min(v)

            #-------------------------------------------------------------------

            return baseline, delta_t

class I_gl_Analyzer(Analyzer):
    def __init__(self, coord_wafer, coord_hicann, save_mean):
        super(I_gl_Analyzer, self).__init__()
        pll_freq = 100e6
        self.logger.INFO("Initializing I_gl_analyzer by measuring ADC sampling frequency.")
        self.trace_averager = createTraceAverager(coord_wafer, coord_hicann)
        self.logger.INFO("TraceAverager created with ADC frequency {} Hz.".format(self.trace_averager.adc_freq))
        self.dt = 129 * 4 * 16 / pll_freq
        # TODO implement different capacitors
        self.C = 2.16456e-12 # Capacitance when bigcap is turned on
        self.save_mean = save_mean

    def __call__(self, t, v, neuron, std, V_rest=None):
        mean_trace, std_trace, n_chunks = self.trace_averager.get_average(v, self.dt)
        t,v = None, None
        mean_trace = np.array(mean_trace)
        # TODO take std of an independent measurement (see CKs method)
        trace_cut, fittimes = self.get_decay_fit_range(mean_trace)
        tau_m, V_rest, height, red_chi2, pcov, infodict = self.fit_exponential(trace_cut, fittimes, std, V_rest, n_chunks)
        if tau_m is not None:
            g_l = self.C / tau_m
        else:
            g_l = None

        result = {"tau_m" : tau_m,
                  "g_l"  : g_l,
                  "reduced_chi2": red_chi2,
                  "V_rest" : V_rest,
                  "height" : height,
                  "pcov"   : pcov,
                  "infodict": infodict,
                  "std" : std,
                  "n_chunks": n_chunks,}
        if self.save_mean:
            result["mean"] = mean_trace
        return result

    def get_decay_fit_range(self, trace, stim_length=65):
        """Cuts the trace for the exponential fit. This is done by calculating the second derivative.
        Returns:
            trace_cut, std_cut, fittimes"""
        filter_width = 250e-9 # Magic number that was carefully tuned to give best results
        dt = 1/self.trace_averager.adc_freq

        stim_steps = int(stim_length * 4 * 16 * self.trace_averager.adc_freq/100e6)
        stim_time = stim_steps * dt

        kernel = scipy.signal.gaussian(len(trace), filter_width / dt)
        kernel /= pylab.sum(kernel)
        kernel = pylab.roll(kernel, int(len(trace) / 2))

        diff  = pylab.roll(trace, -1) - trace
        diff_smooth = pylab.real(pylab.ifft(pylab.fft(kernel) * pylab.fft(diff)))

        diff2_smooth = pylab.roll(diff_smooth, -1) - diff_smooth
        diff2_smooth /= (dt ** 2)

        # For fit start, check if first derivative is really negative.
        # This should make the fit more robust
        diff2_with_negative_diff = np.zeros(len(diff2_smooth))
        for i, j in enumerate(diff_smooth<0):
            if j:
                diff2_with_negative_diff[i] = diff2_smooth[i]

        fitstart = np.argmin(diff2_with_negative_diff)
        fitstop  = np.argmax(diff2_smooth)

        trace = pylab.roll(trace, -fitstart)

        fitstop = fitstop - fitstart
        if fitstop<0:
            fitstop += len(trace)

        fitstart = 0

        trace_cut = trace[fitstart:fitstop]
        fittimes = np.arange(fitstart, fitstop)*dt

        self.logger.TRACE("Cut trace from length {} to length {}.".format(len(trace), len(trace_cut)))

        return trace_cut,fittimes

    def get_initial_parameters(self, times, trace, V_rest):
        height = trace[0] - V_rest
        try:
            # get first element where trace is below 1/e of initial value
            # This does not always work
            tau = times[trace-V_rest < ((trace[0]-V_rest) / np.exp(1))][0]
        except IndexError: # If trace never goes below 1/e, use some high value
            tau = 1e-5
        return [tau, height]
    
    def fit_exponential(self, trace_cut, fittimes, std, V_rest, n_chunks, stim_length = 65):
        """ Fit an exponential function to the mean trace. """
        if V_rest is None:
            def func(t, tau, offset, a):
                return a * np.exp(-(t - t[0]) / tau) + offset
        else:
            def func(t, tau, a):
                return a * np.exp(-(t - t[0]) / tau) + V_rest

        x0 = self.get_initial_parameters(fittimes, trace_cut, V_rest)
        std = std / np.sqrt(n_chunks-1)

        if V_rest is None:
            x0.append(0.7) # If no V_rest is specified, this should be the initial value

        try:
            expf, pcov, infodict, errmsg, ier = curve_fit(
                func,
                fittimes,
                trace_cut,
                x0,
                sigma=std,
                full_output=True)
        except ValueError as e:
            self.logger.WARN("Fit failed: {}".format(e))
            return None, V_rest, None, None, None, None

        tau = expf[0]
        if V_rest == None:
            V_rest = expf[1]
            height = expf[2]
        else:
            height = expf[1]

        DOF = len(fittimes) - len(expf)
        red_chisquare= sum(infodict["fvec"] ** 2) / (DOF)

        self.logger.TRACE("Successful fit: tau={0:.2e} s, V_rest={1:.2f} V, height={2:.2f} V, red_chi2={3:.2f}".format(tau, V_rest, height, red_chisquare))

        return tau, V_rest, height, red_chisquare, pcov, infodict

class V_t_Analyzer(Analyzer):
    def __call__(self, t, v, neuron):

        delta = np.std(v)

        try:
            maxtab, mintab = peakdet(v, delta)
            V_t = np.mean(maxtab[:,1])
        except IndexError as e:
            V_t = np.max(v)

        return {"max" : V_t, "old_max" : np.max(v)}
