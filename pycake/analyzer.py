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

class PeakAnalyzer(Analyzer):
    """ Used to analyze a constant-spiking-pattern (rest > thresh)

        Args:
            t, v, neuron: times, voltage trace, neuron coordinate.
        Returns:
            Result dictionary containing:
                hard_max : max(v)
                hard_min : min(v)
                mean_max : Average maximum of spikes
                mean_min : Average minimum of spikes
                baseline : Baseline (only works with non-zero refractory period)
                frequency: Spike frequency
    """
    def __init__(self):
        super(PeakAnalyzer, self).__init__()

    def __call__(self, t,v, neuron):
        baseline, delta_t = self.find_baseline(t,v)
        mean_max, mean_min = self.get_mean_peak(t, v, neuron)
        freq = self.get_frequency(t, v)
        
        return { "hard_max" : np.max(v),
                 "hard_min" : np.min(v),
                 "mean"     : np.mean(v),
                 "std"      : np.std(v),
                 "mean_max" : mean_max,
                 "mean_min" : mean_min,
                 "baseline" : baseline,
                 "frequency": freq}

    def get_mean_peak(self, t, v, neuron):
        delta = np.std(v)
        try:
            maxtab, mintab = peakdet(v, delta)
            mean_max = np.mean(maxtab[:,1])
            mean_min = np.mean(mintab[:,1])
        except IndexError as e:
            self.logger.INFO("No mean min/max found. Using hard min/max.")
            mean_max = np.max(v)
            mean_min = np.min(v)
        return mean_max, mean_min

    def get_frequency(self, t, v):
        spikes = self.detect_spikes(t,v)
        return self.spikes_to_frequency(spikes)

    def detect_spikes(self, time, voltage):
        """Detect spikes from a voltage trace."""

        # make sure we have numpy arrays
        t = np.array(time)
        v = np.array(voltage)

        # Derivative of voltages
        dv = v[1:] - v[:-1]
        # Take average over 3 to reduce noise and increase spikes
        smooth_dv = dv[:-2] + dv[1:-1] + dv[2:]
        threshhold = -2.5 * np.std(smooth_dv)

        # Detect positions of spikes
        tmp = smooth_dv < threshhold
        pos = np.logical_and(tmp[1:] != tmp[:-1], tmp[1:])
        spikes = t[1:-2][pos]

        return spikes

    def spikes_to_frequency(self, spikes):
        """Calculate the spiking frequency from spikes."""

        # inter spike interval
        isi = spikes[1:] - spikes[:-1]
        if (len(isi) == 0) or (np.mean(isi) == 0):
            return 0
        else:
            return 1./np.mean(isi)

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

class V_reset_Analyzer(PeakAnalyzer):
    """ Uses PeakAnalyzer to get results that are relevant for V_reset

        Returns:
            Result dictionary containing:
                mean_min: mean minimum after spikes.
                baseline: baseline of trace. use only if refractory period is non-zero!
                delta_t : time between spikes
    """
    def __call__(self, t,v, neuron):
        baseline, delta_t = self.find_baseline(t,v)
        mean_max, mean_min = self.get_mean_peak(t, v, neuron)

        return { "mean_min" : mean_min,
                 "baseline" : baseline,
                 "delta_t"  : delta_t}

class V_t_Analyzer(PeakAnalyzer):
    """ Uses PeakAnalyzer to get results that are relevant for V_reset

        Returns:
            Result dictionary containing:
                max: mean maximum of spikes
                old_max: hard maximum of complete trace
    """
    def __call__(self, t,v, neuron):
        mean_max, mean_min = self.get_mean_peak(t, v, neuron)

        return { "max"      : mean_max,
                 "old_max"  : np.max(v)}

class I_gl_Analyzer(Analyzer):
    def __init__(self, coord_wafer, coord_hicann, save_mean=True):
        super(I_gl_Analyzer, self).__init__()
        pll_freq = 100e6
        self.logger.INFO("Initializing I_gl_analyzer by measuring ADC sampling frequency.")
        self.trace_averager = createTraceAverager(coord_wafer, coord_hicann)
        self.logger.INFO("TraceAverager created with ADC frequency {} Hz.".format(self.trace_averager.adc_freq))
        self.dt = 129 * 4 * 16 / pll_freq
        # TODO implement different capacitors
        self.C = 2.16456e-12 # Capacitance when bigcap is turned on
        self.save_mean = save_mean

    def __call__(self, t, v, neuron, std, V_rest_init=0.7, used_current=None):
        mean_trace, std_trace, n_chunks = self.trace_averager.get_average(v, self.dt)
        t,v = None, None
        mean_trace = np.array(mean_trace)
        # TODO take std of an independent measurement (see CKs method)
        trace_cut, fittimes = self.get_decay_fit_range(mean_trace)
        tau_m, V_rest, height, red_chi2, pcov, infodict, ier = self.fit_exponential(trace_cut, fittimes, std, V_rest_init, n_chunks)
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
                  "std" : std,
                  "n_chunks": n_chunks,
                  "used_current": used_current,
                  "ier": ier}
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
        return [tau, height, V_rest]

    def fit_exponential(self, trace_cut, fittimes, std, V_rest_init, n_chunks, stim_length = 65):
        """ Fit an exponential function to the mean trace. """
        def func(t, tau, height, V_rest):
            return height * np.exp(-(t - t[0]) / tau) + V_rest

        x0 = self.get_initial_parameters(fittimes, trace_cut, V_rest_init)
        std = std / np.sqrt(n_chunks-1)

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
            return None, V_rest_init, None, None, None, None, 0

        tau    = expf[0]
        height = expf[1]
        V_rest = expf[2]

        DOF = len(fittimes) - len(expf)
        red_chisquare= sum(infodict["fvec"] ** 2) / (DOF)

        self.logger.TRACE("Successful fit: tau={0:.2e} s, V_rest={1:.2f} V, height={2:.2f} V, red_chi2={3:.2f}".format(tau, V_rest, height, red_chisquare))

        return tau, V_rest, height, red_chisquare, pcov, infodict, ier

V_syntci_psp_max_Analyzer = MeanOfTraceAnalyzer
V_syntcx_psp_max_Analyzer = MeanOfTraceAnalyzer

class I_pl_Analyzer(Analyzer):
    def __call__(self, t, v, neuron):

        delta = np.std(v)

        try:
            maxtab, mintab = peakdet(v, delta)
            maxs = maxtab[:,0]
            #print maxs
            mean = np.mean([x - maxs[i - 1] for i, x in enumerate(maxs[1:])])
            #print mean
            tau_refrac = t[mean]-t[0]
            #print tau_refrac
        except Exception as e:
            print e
            tau_refrac = None

        return {"tau_refrac" : tau_refrac}