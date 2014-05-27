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
        self.dt = 129 * 4 * 16 / pll_freq
        # TODO implement different capacitors
        self.C = 2.16456e-12 # Capacitance when bigcap is turned on
        self.save_mean = save_mean

    def __call__(self, t, v, neuron):
        mean_trace, std_trace, n_mean = self.trace_averager.get_average(v, self.dt)
        mean_trace = np.array(mean_trace)
        # TODO take std of an independent measurement (see CKs method)
        std_trace = np.array(std_trace)
        std_trace /= np.sqrt(np.floor(len(v)/len(mean_trace)))
        tau_m, red_chi2, offset = self.fit_exponential(mean_trace, std_trace)
        if tau_m is not None:
            g_l = self.C / tau_m
        else:
            return None

        result = { "tau_m" : tau_m,
                 "g_l"  : g_l,
                 "reduced_chi2": red_chi2,
                 "offset" : offset}
        if self.save_mean:
            result["mean"] = mean_trace
        return result

    def get_decay_fit_range(self, trace):
        """Cuts the trace for the exponential fit. This is done by calculating the second derivative."""
        filter_width = 250e-9 # Magic number that was carefully tuned to give best results
        dt = 1/self.trace_averager.adc_freq
    
        diff = pylab.roll(trace, 1) - trace
        diff2 = diff - pylab.roll(diff, -1)
        diff2 /= (dt ** 2)
    
        kernel = scipy.signal.gaussian(len(trace), filter_width / dt)
        kernel /= pylab.sum(kernel)
        kernel = pylab.roll(kernel, int(len(trace) / 2))
        diff2_smooth = pylab.real(pylab.ifft(pylab.fft(kernel) * pylab.fft(diff2)))

        fitstart = np.argmin(diff2_smooth)
        fitstop = np.argmax(diff2_smooth)

        if fitstart > fitstop:
            trace = pylab.roll(trace, -fitstart)
            trace_cut = trace[:fitstop-fitstart+len(trace)]
            fittime = np.arange(0, fitstop-fitstart+len(trace))
        else:
            trace_cut = trace[fitstart:fitstop]
            fittime = np.arange(fitstart, fitstop)

        return trace_cut, fittime
    
    def fit_exponential(self, mean_trace, std_trace, stim_length = 65):
        """ Fit an exponential function to the mean trace. """
        def func(x, tau, offset, a):
            return a * np.exp(-(x - x[0]) / tau) + offset
    
        trace_cut, fittime = self.get_decay_fit_range(mean_trace)

        try:
            expf, pcov, infodict, errmsg, ier = curve_fit(
                func,
                fittime,
                trace_cut,
                [.5, 100., 0.1],
                sigma=std_trace[fittime],
                full_output=True)
        except ValueError as e:
            self.logger.WARN("Fit failed: {}".format(e))
            return None, None, None

        tau = expf[0] / self.trace_averager.adc_freq

        DOF = len(fittime) - len(expf)
        red_chisquare = sum(infodict["fvec"] ** 2) / (DOF)

        return tau, red_chisquare, expf[1]

class V_t_Analyzer(Analyzer):
    def __call__(self, t, v, neuron):

        delta = np.std(v)

        try:
            maxtab, mintab = peakdet(v, delta)
            V_t = np.mean(maxtab[:,1])
        except IndexError as e:
            V_t = np.max(v)

        return {"max" : V_t, "old_max" : np.max(v)}

V_syntci_psp_max_Analyzer = MeanOfTraceAnalyzer
V_syntcx_psp_max_Analyzer = MeanOfTraceAnalyzer
