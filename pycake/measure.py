import numpy as np
import pylogging
from Coordinate import NeuronOnHICANN, FGBlockOnHICANN, iter_all
from pyhalbe.HICANN import neuron_parameter, shared_parameter
import pyhalbe

from helpers.WorkerPool import WorkerPool
from helpers.TracesOnDiskDict import TracesOnDiskDict

import time
import os

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
        self.spikes = {}
        self.done = False
        self.traces = None

        if readout_shifts is None:
            self.logger.WARN("No readout shifts found. Shifts are set to 0")
            self.readout_shifts = no_shift
        else:
            self.readout_shifts = ReadoutShift(readout_shifts)

    def save_traces(self, storage_path):
        """Enabling saving of traces to the given file"""
        dirname, filename = os.path.split(storage_path)
        if self.traces is None:
            self.logger.info("Storing traces at '{}'".format(storage_path))
            self.traces = TracesOnDiskDict(dirname, filename)
        else:
            self.logger.warn("traces are already stored, ignoring second call")

    def update_traces_folder(self, folder):
        """Updates the folder the traces are stored. This might be required
        after unpickling"""
        if not self.traces is None:
            self.traces.update_directory(folder)

    def finish(self):
        """ Finish the measurement.
            This sets the 'done' flag and time_finished.
        """
        self.done = True
        self.time_finished = time.asctime()
        if not self.traces is None:
            self.traces.close()

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

    def get_parameters(self, neuron, parameters):
        """ Read out parameters for the given neuron. Shared parameters are
        read from the corresponding FGBlock.

            Args:
                neuron: parameters matching this neuron
                parameter: [list] parameters to read (neuron_parameter or
                           shared_parameter)
            Return:
                list: containing parameters in the requested order
        """
        fgs = self.sthal.hicann.floating_gates
        values = []
        for parameter in parameters:
            if isinstance(parameter, neuron_parameter):
                values.append(fgs.getNeuron(neuron, parameter))
            elif isinstance(parameter, shared_parameter):
                block = neuron.sharedFGBlock()
                values.append(fgs.getShared(block, parameter))
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
        if self.traces is None:
            return None
            # TODO rethink raise KeyError(neuron)

        t, v = self.traces[neuron]
        if apply_readout_shift:
            return t, self.readout_shifts(neuron, v)
        else:
            return t, v

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
            times, trace = self.sthal.read_adc()
            worker.do(neuron, times, self.readout_shifts(neuron, trace), neuron)
            if not self.traces is None:
                self.traces[neuron] = np.array([times, trace])
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
    def __init__(self, sthal, neurons, readout_shifts=None, current=35):
        super(I_gl_Measurement, self).__init__(sthal, neurons, readout_shifts)
        self.currents = current

    def set_current(self, current):
        """ Change current.

            Args:
                current: value in nA
        """
        stim_current = current
        pulse_length = 15
        stim_length = 65

        pll_freq = self.sthal.getPLL()
        self.stimulus_length = (pulse_length+1) * stim_length * 129 / pll_freq

        stimulus = pyhalbe.HICANN.FGStimulus()
        stimulus.setPulselength(pulse_length)
        stimulus.setContinuous(True)

        stimulus[:stim_length] = [stim_current] * stim_length
        stimulus[stim_length:] = [0] * (len(stimulus) - stim_length)

        self.sthal.set_current_stimulus(stimulus)

    def measure_V_rest(self, neuron):
        self.pre_measure(neuron, current=0)
        times, trace = self.sthal.read_adc()
        V_rest = np.mean(trace)
        self.logger.TRACE("Measured V_rest of neuron {0}: {1:.3f} V".format(neuron, V_rest))
        return V_rest

    def _measure(self, analyzer):
        """ Measure traces and correct each value for readout shift.
            Also applies analyzer to measurement
        """
        self.logger.INFO("Measuring.")
        worker = WorkerPool(analyzer)
        for neuron in self.neurons:
            V_rest = self.measure_V_rest(neuron)
            if not self.traces is None:
                self.traces[neuron] = []
            self.logger.TRACE("Measuring neuron {} with current {}".format(neuron, current))
            self.pre_measure(neuron, current=self.current)
            times, trace = self.sthal.read_adc()
            worker.do(neuron, times, self.readout_shifts(neuron, trace), neuron, V_rest)
            if not self.traces is None:
                self.traces[neuron].append((current, np.array([times, trace])))
        self.logger.INFO("Wait for analysis to complete.")
        return worker.join()

    def pre_measure(self, neuron, current):
        self.set_current(int(current))
        self.sthal.switch_current_stimulus_and_output(neuron)
