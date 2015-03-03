import numpy as np

import pysthal
import pyhalbe
import pylogging
import Coordinate
import time
from pycake.helpers.sthal import StHALContainer

logger = pylogging.get("pycake.helper.TraceAverager")


def hardware(func):
    """decorator for functions which connect to hardware"""
    return func


class Configurator(pysthal.HICANNConfigurator):
    def hicann_init(self, h):
        pyhalbe.HICANN.init(h, False)

    def config_synapse_array(self, handle, data):
        pass

    def config_floating_gates(self, handle, data):
        pass


def _find_spikes_in_preout(trace):
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


@hardware
def _get_preout_trace(coord_wafer, coord_hicann, bg_rate, recording_time):
    """
    Read preout of upper (debug) synapse driver.

    note: This won't work on Analog 1, because stimulateNeurons doesn't give input
    to the lower synapse driver with debug output.
    """
    analog = Coordinate.AnalogOnHICANN(0)
    sthal = StHALContainer(coord_wafer, coord_hicann, analog, recording_time)
    # We need SynapseDriverOnHICANN(Enum(111)), this should be covered
    sthal.stimulateNeurons(bg_rate, 4)
    sthal.hicann.analog.set_preout(analog)
    # TODO skip floating gates to speed up
    sthal.write_config(configurator=Configurator())
    time.sleep(1)  # Settle driver locking
    times, trace = sthal.read_adc()
    sthal.disconnect()
    return trace


def createTraceAverager(coord_wafer, coord_hicann):
    """
    Creates a TraceAverage by using constant spike input on the upper
    synapse driver preout.
    """
    analog = Coordinate.AnalogOnHICANN(0)
    logger.info("Create TraceAverager for {} on {}.".format(
        analog, coord_hicann))
    bg_rate = 100.0e3
    recording_time = 1000.0 / bg_rate
    trace = _get_preout_trace(coord_wafer, coord_hicann, bg_rate, recording_time)
    pos = _find_spikes_in_preout(trace)
    n = len(pos)
    expected_t = np.arange(n) / bg_rate
    adc_freq, _ = np.polyfit(expected_t, pos, 1)
    if not 95e6 < adc_freq < 97e6:
        raise RuntimeError("Found ADC frequency of {}, this is unlikly".format(
            adc_freq))
    return TraceAverager(adc_freq)


class TraceAverager(object):
    def __init__(self, adc_freq):
        self.adc_freq = adc_freq
        msg = "Initialized TraceAverager with adc frequency of {} MHz"
        logger.INFO(msg.format(self.adc_freq))

    def get_chunks(self, trace, dt):
        """Splits trace in chunks of lenght dt"""
        n = len(trace)
        dpos = dt * self.adc_freq
        chunks = int(np.floor(n / dpos))
        window_size = int(dpos)
        result = np.empty((chunks, window_size))
        # Starting position of chunks
        positions = np.arange(chunks, dtype=np.float) * dpos
        for ii, pos in enumerate(positions):
            result[ii] = trace[pos:pos+window_size]
        return result

    def get_average(self, trace, period):
        """Gives mean and std of trace slices with length period.

        Args:
            voltage trace,
            period

        Returns:
            (averaged trace,
             std of points,
             number of chunks used for average)
        """
        chunks = self.get_chunks(trace, period)
        return np.mean(chunks, axis=0), np.std(chunks, axis=0, ddof=1), chunks.shape[0]

    def get_adc_freq(self):
        return self.adc_freq

    def set_adc_freq(self, freq):
        self.adc_freq = freq
