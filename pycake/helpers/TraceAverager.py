import numpy

from pyhalbe import Coordinate
from sthal import StHALContainer

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
    # We need SynapseDriverOnHICANN(Enum(111)), this should be covered
    sthal.stimulateNeurons(bg_rate, 4)
    sthal.hicann.set_preout(analog)
    # TODO skip floating gates to speed up
    sthal.write_config()

    times, trace = sthal.read_adc()
    pos = _find_spikes_in_preout(trace)

    n = len(pos)
    expected_t = np.arange(n) / bg_rate
    adc_freq, _ = np.polyfit(pos, expected_t, 1)

    sthal.disconnect()

    return TraceAverager(adc_freq)

class TraceAverager(object):
    def __init__(self,  adc_freq):
        self.adc_freq = adc_freq

    def _get_chunks(trace, dt):
        n = len(trace)
        dpos = dt * adc_freq
        window_size = int(dpos)
        pos = 0
        while pos + dpos < n:
            a = int(pos)
            b = a + window_size
            yield trace[a:b]
            pos += dpos
        return

    def get_chunks(trace, dt):
        """Splits trace in chunks of lenght dt"""
        return np.array([x for x in self._get_chunks(trace, dt)])
