import numpy as np
import pylogging
from collections import defaultdict
import pyhalbe
import pycalibtic
from pyhalbe.Coordinate import NeuronOnHICANN, FGBlockOnHICANN
from pyhalbe.HICANN import neuron_parameter, shared_parameter

# Import everything needed for saving:
import pickle
import time

class Measurement(object):
    """ This class takes a sthal container, writes the configuration to
        the hardware and measures traces. Also, the time of measurement is saved.

        Args:
            sthal: fully configured StHALContainer as found in pycake.helpers.sthal
            neurons: list of NeuronOnHICANN coordinates
            readout_shifts (optional): Dictionary {neuron: shift} in V that contains
                the individual readout shifts extracted from V_reset measurements.
                Note: Every voltage value from the trace is shifted back by -shift!
                If None is given, the shifts are set to 0.
    """
    def __init__(self, sthal, neurons, readout_shifts=None): 
        # TODO: callable readout_shifter instead of dict
        # readout_shifter(neuron, trace)
        self.sthal = sthal
        self.neurons = neurons
        self.recording_time = self.sthal.recording_time
        self.time = time.asctime()
        self.traces = {}
        self.spikes = {}

        if readout_shifts is None:
            # TODO use logger
            print "No readout shifts found. Continuing without readout shifts."
            self.readout_shifts = lambda neuron, v: v
        else:
            self.readout_shifts = lambda neuron, v: v - readout_shifts[neuron]

    def done(self):
        return len(self.traces) == len(self.neurons)

    def get_parameter(self, parameter, coords):
        """ Used to read out parameters that were used during this measurement.

            Args: 
                parameter: which parameter (neuron_parameter or shared_parameter)
                coord: pyhalbe coordinate or list of coordinates (FGBlockOnHICANN or NeuronOnHICANN)
            Return:
                DAC value of that parameter
        """
        fgs = self.sthal.hicann.floating_gates
        if not isinstance(coords, list):
            coords = [coords]
        values = {}
        for coord in coords:
            if isinstance(parameter, neuron_parameter) and isinstance(coord, NeuronOnHICANN):
                values[coord] = fgs.getNeuron(coord, parameter)
            elif isinstance(parameter, shared_parameter) and isinstance(coord, FGBlockOnHICANN):
                values[coord] = fgs.getShared(coord, parameter)
            else:
                print "Invalid parameter <-> coordinate pair"
                # TODO raise TypeError
        return values

    def get_trace(self, neuron):
        v = self.traces[neuron]
        return self.readout_shifts(neuron, v)

    def iter_traces(self):
        for neuron in self.traces:
            yield self.get_trace(neuron)
        return

    def configure(self):
        """ Write StHALContainer configuration to the hardware.
            This connects to the hardware.
        """
        self.sthal.write_config()

    def measure(self):
        """ Measure traces and correct each value for readout shift.
            Changes traces to numpy arrays

            This will set:
                self.traces = {neuron: trace}
                self.spikes = {neuron: spikes} #TODO this is not yet implemented
        """
        for neuron in self.neurons:
            self.sthal.switch_analog_output(neuron)
            t, v = self.sthal.read_adc()
            t = np.array(t)
            v = np.array(v)
            self.traces[neuron] = (t, v)
        
    def run_measurement(self):
        """ First configure, then measure
        """
        self.configure()
        self.measure()

        self.sthal.disconnect()

