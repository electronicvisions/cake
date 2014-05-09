import numpy as np
import pylogging
from pyhalbe.Coordinate import NeuronOnHICANN, FGBlockOnHICANN
from pyhalbe.HICANN import neuron_parameter, shared_parameter

from helpers.WorkerPool import WorkerPool

# Import everything needed for saving:
import time

def no_shift(neuron, v):
    return v

class ReadoutShift(object):
    def __init__(self, shifts):
        self.shifts = shifts
    def __call__(self, neuron, v):
        return v - self.shifts[neuron]

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

    logger = pylogging.get("pycake.measurement")

    def __init__(self, sthal, neurons, readout_shifts=None): 
        # TODO: callable readout_shifter instead of dict
        # readout_shifter(neuron, trace)
        self.sthal = sthal
        self.neurons = neurons
        self.time_created = time.time()
        self.traces = {}
        self.spikes = {}
        self.done = False

        if readout_shifts is None:
            self.logger.WARN("No readout shifts found. Shifts are set to 0")
            self.readout_shifts = no_shift
        else:
            self.readout_shifts = ReadoutShift(readout_shifts)

    def finish(self):
        """ Finish the measurement.
            This sets the 'done' flag and time_finished.
        """
        self.done = len(self.traces) == len(self.neurons)
        self.time_finished = time.asctime()

    def clear_traces(self):
        """ Clear all the traces. If traces are already recorded, they are set to None.
            Else, they are set to an empty dict
        """
        if self.done:
            self.traces = {neuron: None for neuron in self.neurons}
        else:
            self.traces = {}
        pass

    def get_parameter(self, parameter, coords):
        """ Used to read out parameters that were used during this measurement.

            Args: 
                parameter: which parameter (neuron_parameter or shared_parameter)
                coord: pyhalbe coordinate or list of coordinates (FGBlockOnHICANN or NeuronOnHICANN)
            Return:
                DAC value(s) of that parameter
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
                return
                # TODO raise TypeError
        return values

    def get_neurons(self):
        return self.neurons

    def get_trace(self, neuron, apply_readout_shift = True):
        """ Get the voltage trace of a neuron.
            Other than in the measurement.trace dictionary,
            these traces are shifted by the readout shift.

            Args:
                neuron: neuron coordinate

            Returns:
                tuple (t, v)
        """
        t, v = self.traces[neuron]
        t = np.array(t)
        v = np.array(v)
        return t, self.readout_shifts(neuron, v)

    def iter_traces(self):
        for neuron in self.traces:
            t, v = self.get_trace(neuron)
            yield t, v, neuron
        return

    def configure(self):
        """ Write StHALContainer configuration to the hardware.
            This connects to the hardware.
        """
        self.sthal.write_config()

    def pre_measure(self, neuron):
        self.sthal.switch_analog_output(neuron)

    def _measure(self, analyzer):
        """ Measure traces and correct each value for readout shift.
            Changes traces to numpy arrays

            This will set:
                self.traces = {neuron: trace}
                self.spikes = {neuron: spikes} #TODO this is not yet implemented
        """
        self.logger.INFO("Measuring.")
        worker = WorkerPool(analyzer)
        for neuron in self.neurons:
            self.pre_measure(neuron)
            t, v = self.sthal.read_adc()
            t = np.array(t)
            v = np.array(v)
            self.traces[neuron] = (t, v)
            # apply readout shift
            t, v = self.get_trace(neuron)
            worker.do(neuron, t, v, neuron)
        self.logger.INFO("Wait for analysis to complete.")
        return worker.join()

    def run_measurement(self, analyzer):
        """ First configure, then measure
        """
        self.logger.INFO("Connecting to hardware and configuring.")
        self.configure()
        result = self._measure(analyzer)
        self.logger.INFO("Measurement done, disconnecting from hardware.")
        self.finish()
        self.sthal.disconnect()
        return result

class I_gl_Measurement(Measurement):
    def pre_measure(self, neuron):
        self.sthal.switch_current_stimulus_and_output(neuron)

