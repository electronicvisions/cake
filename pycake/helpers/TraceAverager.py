import numpy as np

from pyhalbe import Coordinate
from pycake.helpers.sthal import StHALContainer

import pylogging

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.signal
import pylab

def _find_spikes_in_preout(trace):
    """Detects spikes in a trace of the HICANN preout

    The signal of the preout seems to be usually quite strong.
    """
    th = 0.5
    tmp = np.where((trace >= th)[:-1] != (trace >= th)[1:])[0]
    tmp = tmp[:len(tmp)/2*2]
    spike_pos = tmp.reshape((len(tmp)/2,2))
    positions = []
    for begin, end in spike_pos:
        begin, end = begin - 1, end + 1
        t = np.arange(begin, end)
        pos = np.dot(trace[begin:end], t) / np.sum(trace[begin:end])
        positions.append(pos)
    return np.array(positions)

def createTraceAverager(coord_wafer, coord_hicann):
    analog = Coordinate.AnalogOnHICANN(0)
    bg_rate = 100.0e3
    recording_time = 1000.0 / bg_rate

    sthal = StHALContainer(coord_wafer, coord_hicann, analog, recording_time)
    pylogging.set_loglevel(pylogging.get("pycake.helper.sthal"), pylogging.LogLevel.DEBUG)
    # We need SynapseDriverOnHICANN(Enum(111)), this should be covered
    sthal.stimulateNeurons(bg_rate, 4)
    sthal.hicann.analog.set_preout(analog)
    # TODO skip floating gates to speed up
    sthal.write_config()

    times, trace = sthal.read_adc()
    pos = _find_spikes_in_preout(trace)

    n = len(pos)
    
    expected_t = np.arange(n) / bg_rate
    adc_freq, _ = np.polyfit(expected_t, pos, 1)

    sthal.disconnect()

    return TraceAverager(adc_freq)

class TraceAverager(object):
    def __init__(self,  adc_freq):
        self.adc_freq = adc_freq

    def _get_chunks(self, trace, dt):
        n = len(trace)
        dpos = dt * self.adc_freq
        window_size = int(dpos)
        pos = 0
        while pos + dpos < n:
            a = int(pos)
            b = a + window_size
            yield trace[a:b]
            pos += dpos
        return

    def get_chunks(self, trace, dt):
        """Splits trace in chunks of lenght dt"""
        return np.array([x for x in self._get_chunks(trace, dt)])

    def get_average(self, trace, dt):
        """Gives mean and std of trace slices with length dt"""
        chunks = self.get_chunks(trace, dt)
        return np.mean(chunks, axis = 0), np.std(chunks, axis = 0, ddof = 1)

    def get_decay_fit_range(self, trace):
        """Cuts the trace for the exponential fit. This is done by calculating the second derivative."""
        filter_width = 100e-9
        dt = 1/self.adc_freq
    
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
            trace = pylab.roll(trace, fitstart - fitstop)
            trace_cut = trace[fitstop:fitstart]
            fittime = np.arange(fitstop, fitstart)
        else:
            trace_cut = trace[fitstart:fitstop]
            fittime = np.arange(fitstart, fitstop)

        return trace_cut, fittime
    
    def fit_exponential(self, mean_trace, stim_length = 65):
        """ Fit an exponential function to the mean trace. """
        func = lambda x, tau, offset, a: a * np.exp(-(x - x[0]) / tau) + offset
    
        trace_cut, fittime = self.get_decay_fit_range(mean_trace)
    
        fit = curve_fit(
            func,
            fittime,
            trace_cut,
            [.5, 100., 0.1])
    
        tau = fit[0][0] / self.adc_freq * 1e6
    
        return tau



