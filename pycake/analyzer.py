import numpy as np
np.seterr(all='raise')
np.seterr(under='warn')
import pylogging
from scipy.integrate import simps
from pycake.helpers.peakdetect import peakdet
from pycake.logic.spikes import spikes_to_frequency
from pycake.logic.exponential_fit import fit_exponential, get_decay_fit_range, failed_dict
from pycake.helpers.TraceAverager import TraceAverager

from sims.sim_denmem_lib import NETS_AND_PINS


class Analyzer(object):
    """ Takes a measurement and analyses it.
    """
    logger = pylogging.get("pycake.analyzer")

    def __call__(self, neuron, **traces):
        """ Returns a dictionary of results:
            {neuron: value}
        """
        raise NotImplementedError("Not implemented in {}".format(
            type(self).__name__))


class MeanOfTraceAnalyzer(Analyzer):
    """ Analyzes traces for E_l measurement.
    """
    def __call__(self, neuron, t, v, **traces):
        spike_counter = traces.get(NETS_AND_PINS.SpikeCounter, None)
        spikes = -1.0 if spike_counter is None else spike_counter[-1]

        if spikes > 0.0:
            return {
                "spikes": spikes,
            }
        else:
            return {
                "spikes": spikes,
                "mean": np.mean(v),
                "std": np.std(v),
                "max": np.max(v),
                "min": np.min(v)
            }


class PeakAnalyzer(Analyzer):
    """ Used to analyze a constant-spiking-pattern (rest > thresh)

        Init Args:
            analyze_slopes: if True, also analyzes the slopes between spikes
                            This takes more time, so deactivate if not needed
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
    def __init__(self, analyze_slopes=False):
        super(PeakAnalyzer, self).__init__()
        self.analyze_slopes = analyze_slopes

    def __call__(self, neuron, t, v, **traces):
        maxtab, mintab = self.get_peaks(t, v)
        mean_max, mean_min, std_max, std_min = self.get_mean_peak(t, v, maxtab, mintab)
        maxindex = maxtab[:, 0]
        spikes = t[maxindex]
        freq = spikes_to_frequency(spikes)
        try:
            mean_dt = 1./freq
        except ZeroDivisionError:
            self.logger.WARN("Division by zero -> dt set to inf")
            mean_dt = np.Inf

        results = {"hard_max": np.max(v),
                   "hard_min": np.min(v),
                   "mean": np.mean(v),
                   "std": np.std(v),
                   "mean_max": mean_max,
                   "mean_min": mean_min,
                   "frequency": freq,
                   "mean_dt": mean_dt,
                   "maxtab": maxtab,
                   "mintab": mintab}

        if self.analyze_slopes:
            slopes_rising, slopes_falling = self.get_slopes(t, v, maxtab, mintab)
            results['slopes_rising'] = slopes_rising
            results['slopes_falling'] = slopes_falling

        return results

    def get_peaks(self, t, v):
        delta = np.std(v)
        return peakdet(v, delta)

    def get_slopes(self, t, v, maxtab, mintab):
        """ Returns all the slopes of a trace

            Return:
                slopes_rising, slopes_falling
        """
        imin = np.array(mintab[:, 0], dtype=int)
        valmin = mintab[:, 1]
        imax = np.array(maxtab[:, 0], dtype=int)
        valmax = maxtab[:, 1]
        dv_down = []
        dt_down = []
        dt_up = []
        dv_up = []

        for i in range(min(len(valmax), len(valmin))):
            dv_down.append(valmax[i] - valmin[i])
            dt_down.append(t[imax[i]] - t[imin[i]])
        for i in range(min(len(valmax), len(valmin))-1):
            dv_up.append(valmax[i+1] - valmin[i])
            dt_up.append(t[imax[i+1]] - t[imin[i]])

        dv_down = np.array(dv_down)
        dt_down = np.array(dt_down)
        dv_up = np.array(dv_up)
        dt_up = np.array(dt_up)
        slopes_falling = dv_down/dt_down
        slopes_rising = dv_up/dt_up

        return slopes_rising, slopes_falling

    def get_mean_peak(self, t, v, maxtab, mintab):
        try:
            mean_max = np.mean(maxtab[:, 1])
            std_max = np.std(maxtab[:, 1])
            mean_min = np.mean(mintab[:, 1])
            std_min = np.std(mintab[:, 1])
        except IndexError:
            self.logger.INFO("No mean min/max found. Using hard min/max.")
            mean_max = np.max(v)
            mean_min = np.min(v)
            std_max = 0
            std_min = 0
        return mean_max, mean_min, std_max, std_min

    def get_mean_dt(self, t, v, maxtab, mintab):
        spike_indices = np.array(maxtab[:, 0], dtype=int)
        spiketimes = t[spike_indices]
        dts = np.roll(spiketimes, -1) - spiketimes
        dts = dts[0:-1]
        return np.mean(dts)

    def find_baseline(self, t, v):
            """ find baseline of trace

                t - list of time stamps
                v - corresponding values

                returns baseline and time between values in unit index (do t[delta_t] to get delta in units of time)
            """

            try:

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

            except FloatingPointError as e:
                baseline = np.min(v)
                delta_t = 0
                self.logger.WARN("Baseline finding failed because of {}. Returning minimum of trace: {}.".format(e, baseline))

            #-------------------------------------------------------------------

            return baseline, delta_t


class V_convoff_Analyzer(Analyzer):
    """
    Averaging analyzser
    """
    NET_SYN_OTA_OUTPUT = ""
    NET_SYN_CURRENT = ""

    def __init__(self, spiketimes):
        self.spiketimes = spiketimes

    def __call__(self, neuron, t, v, **traces):

        spike_counter = traces.get(NETS_AND_PINS.SpikeCounter, None)
        spikes = -1.0 if spike_counter is None else spike_counter[-1]

        if spikes > 0.0:
            return {
                "spikes": spikes
            }

        t0 = t.searchsorted(0.9 * self.spiketimes[0])
        baseline = np.mean(v[:t0])
        baseline_std = np.std(v[:t0])
        area = simps(v - baseline, t)

        result = {
            "t0": t0,
            "stim_spikes": self.spiketimes,
            "baseline": baseline,
            "baseline_std": baseline_std,
            "psp_area": area,
            "spikes": spikes,
        }

        ota_current = traces.get(self.NET_SYN_OTA_OUTPUT, None)
        if ota_current is not None:
            result["ota_resting_current"] = np.mean(ota_current[:t0])
        syn_current = traces.get(self.NET_SYN_CURRENT, None)
        if syn_current is not None:
            result["syn_resting_current"] = np.mean(syn_current[:t0])

        return result


class V_convoffi_Analyzer(V_convoff_Analyzer):
    NET_SYN_OTA_OUTPUT = NETS_AND_PINS.I_syni_ota_output
    NET_SYN_CURRENT = NETS_AND_PINS.I_membrane_syni


class V_convoffx_Analyzer(V_convoff_Analyzer):
    NET_SYN_OTA_OUTPUT = NETS_AND_PINS.I_synx_ota_output
    NET_SYN_CURRENT = NETS_AND_PINS.I_membrane_synx


class V_reset_Analyzer(PeakAnalyzer):
    """ Uses PeakAnalyzer to get results that are relevant for V_reset

        Returns:
            Result dictionary containing:
                mean_min: mean minimum after spikes.
                baseline: baseline of trace. use only if refractory period is non-zero!
                delta_t : time between spikes
    """
    def __call__(self, neuron, t, v, **traces):
        baseline, delta_t = self.find_baseline(t, v)
        maxtab, mintab = self.get_peaks(t, v)
        mean_max, mean_min, std_max, std_min = self.get_mean_peak(t, v, maxtab, mintab)

        return {"mean_min": mean_min,
                "mean_max": mean_max,
                "baseline": baseline,
                "delta_t": delta_t}


class V_t_Analyzer(PeakAnalyzer):
    """ Uses PeakAnalyzer to get results that are relevant for V_reset

        Returns:
            Result dictionary containing:
                max: mean maximum of spikes
                old_max: hard maximum of complete trace
    """
    def __call__(self, neuron, t, v, **traces):
        maxtab, mintab = self.get_peaks(t, v)
        mean_max, mean_min, std_max, std_min = self.get_mean_peak(t, v, maxtab, mintab)

        return {"max": mean_max,
                "std_max": std_max,
                "old_max": np.max(v)}


class I_gl_Analyzer(Analyzer):
    def __init__(self, pll_freq=100e6):
        super(I_gl_Analyzer, self).__init__()
        self.logger.INFO("I_gl_Analyzer is using pll={}".format(pll_freq))

        # Create trace_averager with empty ADC frequency.
        self.trace_averager = TraceAverager(None)

        # stimulation period
        # FG cell count * 1/(1/4) (pll_freq divider) * (pulse_length + 1)
        self.dt = 129 * 4 * 16 / pll_freq

        # TODO implement different capacitors
        self.C = 2.16456e-12  # Capacitance when bigcap is turned on
        self.logger.INFO("Initializing I_gl_analyzer using C={} (bigcap).".format(self.C))

    def set_adc_freq(self, freq):
        self.trace_averager.set_adc_freq(freq)

    def __call__(self, neuron, t, v, std, current=None, save_mean=False, **traces):
        # average over all periods, reduce to one smooth period
        if 'adc_freq' in traces:
            self.set_adc_freq(traces['adc_freq'])
        mean_trace, std_trace, n_chunks = self.trace_averager.get_average(v, self.dt)

        # edge detection of decay,
        cut_t, cut_v = get_decay_fit_range(mean_trace, self.trace_averager.adc_freq, fwidth=64)

        if len(cut_v) < (len(mean_trace)/4):
            # sanity check: if cut trace too short, something failed
            # return empty data and error
            result = failed_dict({'error': "Cut trace too short"})
        else:
            result = fit_exponential(cut_t, cut_v, std, n_chunks=n_chunks, full_output=True)

        if result['ier'] > 0:
            # if fit was succesful, do not save infodict
            result['infodict'] = None

        if save_mean:
            result["mean"] = mean_trace

        if current is not None:
            result['current'] = current

        return result


class I_gl_sim_Analyzer(I_gl_Analyzer):
    pass


V_syntci_psp_max_Analyzer = MeanOfTraceAnalyzer
V_syntcx_psp_max_Analyzer = MeanOfTraceAnalyzer


class ISI_Analyzer(Analyzer):
    """Calculate ISIs from trace and their standard deviation.

    Neuron is set to constant-spiking state with constant refractory period.
    Afterwards, neuron is set to constant-spiking state without refractory
    period.

    Refractory time is measured as difference between ISI with and without refractory
    period.

    Please note that mean_reset_time only gives valid results if refractory period is very small
    """

    def __call__(self, neuron, t, v, **traces):
        delta = np.std(v)
        maxtab, mintab = peakdet(v, delta)

        max_idx = maxtab[:, 0].astype(int)  # indices of maxima
        spike_times = t[max_idx]
        isi = np.diff(spike_times)
        mean_isi = np.mean(isi)
        std_isi = np.std(isi)
        mean_max = np.mean(maxtab[:,1])
        mean_min = np.mean(mintab[:,1])
        amplitude = mean_max - mean_min

        dt = np.mean(t[1:] - t[:-1])
        l = len(mintab)
        mean_reset_time = abs(np.mean(mintab[:,0][:l] - maxtab[:,0][:l]) * dt)

        return {"mean_isi": mean_isi,
                "std_isi": std_isi,
                "amplitude": amplitude,
                "mean_reset_time": mean_reset_time
                }


class Spikes_Analyzer(Analyzer):
    def __call__(self, spikes, neuron):

        n_spikes = len(spikes)

        if n_spikes > 1:
            isis = np.diff(spikes)
            try:
                mean_isi = np.mean(isis)
            except Exception as e:
                print e
                print spikes
                print isis
                raise(e)
        else:
            mean_isi = None

        return {"spikes_mean_isi": mean_isi,
                "spikes_n_spikes": n_spikes}


class ADCFreq_Analyzer(Analyzer):
    def __call__(self, t, v, bg_rate):
        """Detects spikes in a trace of the HICANN preout

        The signal of the preout seems to be usually quite strong.
        """
        pos = self._find_spikes_in_preout(v)
        n = len(pos)
        expected_t = np.arange(n) / bg_rate
        adc_freq, _ = np.polyfit(expected_t, pos, 1)
        if not 95e6 < adc_freq < 97e6:
            raise RuntimeError("Found ADC frequency of {}, this is unlikly".format(
                adc_freq))
        return {"adc_freq": adc_freq}

    def _find_spikes_in_preout(self, trace):
        """Detects spikes in a trace of the HICANN preout

        The signal of the preout seems to be usually quite strong.
        """
        th = 0.5
        tmp = np.where((trace >= th)[:-1] != (trace >= th)[1:])[0]
        tmp = tmp[:len(tmp)/2*2]
        spike_pos = tmp.reshape((len(tmp)/2, 2))
        positions = []
        for begin, end in spike_pos:
            begin, end = begin - 1, end + 1
            t = np.arange(begin, end)
            pos = np.dot(trace[begin:end], t) / np.sum(trace[begin:end])
            positions.append(pos)
        return np.array(positions)
