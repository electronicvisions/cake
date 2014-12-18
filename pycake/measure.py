import numpy as np
import pylogging
from Coordinate import NeuronOnHICANN, FGBlockOnHICANN, iter_all, GbitLinkOnHICANN
import Coordinate
from pyhalbe.HICANN import neuron_parameter, shared_parameter
import pyhalbe
import pysthal
import pycake

from helpers.WorkerPool import WorkerPool
from helpers.TracesOnDiskDict import TracesOnDiskDict

import time
import os

from collections import defaultdict

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
        for neuron in neurons:
            if not isinstance(neuron, NeuronOnHICANN):
                raise TypeError("Expected list of integers")
        # TODO: callable readout_shifter instead of dict
        # readout_shifter(neuron, trace)
        self.sthal = sthal
        self.neurons = neurons
        self.time_created = time.time()
        self.spikes = {}
        self.done = False
        self.traces = None
        # Debug for repeated ADC traces
        self.last_trace = None
        self.adc_status = []


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
                block = neuron.toSharedFGBlockOnHICANN()
                values.append(fgs.getShared(block, parameter))
            else:
                raise TypeError("invalid parameter type {}".format(type(parameter)))
        return values

    def get_neurons(self):
        return self.neurons

    def get_trace(self, neuron, apply_readout_shift=True):
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
        """
        Iterate over neuron traces

        Returns:
            ([numpy array] times, [numpy array] voltages [mV],
             [NeuronOnHICANN] neuronId)
        """
        for neuron in self.traces:
            t, v = self.get_trace(neuron)
            yield t, v, neuron
        return

    def configure(self, configurator):
        """ Write StHALContainer configuration to the hardware.
            This connects to the hardware.
        """
        self.sthal.write_config(configurator=configurator)

    def pre_measure(self, neuron, l1address=None):
        self.sthal.switch_analog_output(neuron, l1address=l1address)

    def get_readout_hicann(self):

        readout_wafer = pysthal.Wafer()
        HRC = pysthal.HICANNReadoutConfigurator(readout_wafer)
        self.sthal.wafer.configure(HRC)
        readout_hicann = readout_wafer[self.sthal.coord_hicann]

        return readout_hicann

    def get_spikes(self, nbs):
        """

        nbs: neuron blocks to be enabled

        """

        print "enabling neurons on nbs {}".format(nbs)

        self.sthal.wafer.clearSpikes()

        spikes = pysthal.Vector_Spike()
        for t in [1e-3]:
            spikes.append(pysthal.Spike(pyhalbe.HICANN.L1Address(1), t))
        sending_link = Coordinate.GbitLinkOnHICANN(0)
        self.sthal.hicann.sendSpikes(sending_link, spikes)

        #shuffled_neurons = self.neurons[:]
        #random.shuffle(shuffled_neurons)

        self.sthal.hicann.disable_aout()
        self.sthal.hicann.disable_l1_output()
        self.sthal.hicann.disable_firing()

        activated_neurons = []

        for n, neuron in enumerate(self.neurons):

            for i in nbs:

                if neuron.id().value() >= 32*i and neuron.id().value() < 32*(i+1):

                    nonb = neuron.toNeuronOnNeuronBlock()
                    addr = neuron.toNeuronOnNeuronBlock().x().value() + neuron.toNeuronOnNeuronBlock().y().value()*32

                    #print "enable", neuron, addr

                    self.sthal.hicann.enable_l1_output(neuron, pyhalbe.HICANN.L1Address(addr))
                    self.sthal.hicann.neurons[neuron].activate_firing(True)

                    activated_neurons.append(neuron)

        for channel in Coordinate.iter_all(Coordinate.GbitLinkOnHICANN):
            self.sthal.hicann.layer1[channel] = pyhalbe.HICANN.GbitLink.Direction.TO_DNC

        #self.sthal.wafer.configure(pycake.helpers.sthal.UpdateAnalogOutputConfigurator())
        self.sthal.wafer.configure(pycake.helpers.sthal.FooHICANNConfigurator())
        #self.sthal.wafer.configure(pysthal.DontProgramFloatingGatesHICANNConfigurator())

        runner = pysthal.ExperimentRunner(0.1e-3)
        self.sthal.wafer.start(runner)

        no_spikes = []
        spiketimes = []
        spikeaddresses = []

        addr_spikes = defaultdict(list)

        for ii, channel in enumerate(iter_all(GbitLinkOnHICANN)):
            received = self.sthal.hicann.receivedSpikes(channel)
            times, addresses = received.T

            print channel, len(received)

            no_spikes.append(len(received))
            spiketimes.append(times)
            spikeaddresses.append(addresses)

            for t, a in zip(times, addresses):
                if a < 32:
                    n = a + ii*32
                else:
                    n = 256 + (a-32) + ii*32

                #print n, a, ii
                addr_spikes[Coordinate.NeuronOnHICANN(Coordinate.Enum(int(n)))].append(t)

        for neuron, spikes in addr_spikes.iteritems():

            if neuron in activated_neurons:
                self.spikes[neuron] = spikes

            if neuron not in activated_neurons and len(spikes) != 0:
                print "neuron {} was not activated, but spiked {} times".format(neuron, len(spikes))

            #print neuron, len(spikes)

        #print self.get_readout_hicann().layer1

    def _measure(self, analyzer):
        """ Measure traces and correct each value for readout shift.
            Changes traces to numpy arrays

            This will set:
                self.traces = {neuron: trace}
                self.spikes = {neuron: spikes} #TODO this is not yet implemented
        """

        self.last_trace = np.array([])
        self.adc_status = []
        self.logger.INFO("Measuring.")

        #print "at start of _measure"
        #print self.get_readout_hicann().layer1.getMergerTree()

        """
        with WorkerPool(analyzer) as worker:
            for n, neuron in enumerate(self.neurons):

                #print "right before pre_measure"
                #print self.get_readout_hicann().layer1.getMergerTree()

                self.pre_measure(neuron)
                #print "after pre_measure"
                #print self.get_readout_hicann().layer1.getMergerTree()
                times, trace = self.sthal.read_adc()
                worker.do(
                    neuron, times, self.readout_shifts(neuron, trace), neuron)
                if not self.traces is None:
                    self.traces[neuron] = np.array([times, trace])
                # DEBUG stuff
                self.adc_status.append(self.sthal.read_adc_status())
                if np.array_equal(trace, self.last_trace):
                    self.logger.ERROR(
                        "ADC trace didn't change from the last "
                        "readout, printing status information of all ADC "
                        "previous readouts:\n" + "\n".join(self.adc_status))
                    raise RuntimeError(
                        "Broken ADC readout abort measurement (details see "
                        "log messages)")
                self.last_trace = trace
                # DEBUG stuff end

                # SPIKES
                # Collect data

            self.last_trace = None

            self.logger.INFO("Wait for analysis to complete.")
        """

        for nb in xrange(0,16):
            self.get_spikes([nb])

        for neuron, spikes in self.spikes.iteritems():
            print neuron, len(spikes)

        #for nb in xrange(0,17):
        #    self.get_spikes(range(nb))

        #return worker.join()

    def run_measurement(self, analyzer, configurator=None):
        """ First configure, then measure
        """
        self.logger.INFO("Connecting to hardware and configuring.")
        self.configure(configurator)
        result = self._measure(analyzer)
        self.logger.INFO("Measurement done, disconnecting from hardware.")
        self.finish()
        self.sthal.disconnect()
        return result


class I_gl_Measurement(Measurement):
    """ This measurement is done in three steps:

        1) Measure V_rest without injecting current
        2) Vary input current from 1 to 30 nA to determine best value for fit
        3) Measure longer trace that is used for the fit

        This takes a lot of time!

        skip_I_gl can be used to skip step 2) above a certain I_gl (default 700 nA,
        adjust this for other SpeedUp factors!)
    """
    def __init__(self, sthal, neurons, readout_shifts=None, currents=range(30), skip_I_gl=2000):
        super(I_gl_Measurement, self).__init__(sthal, neurons, readout_shifts)
        self.currents = currents
        self.skip_I_gl = skip_I_gl

    def measure_V_rest(self, neuron):
        """ Measure V_rest and independend std by injecting 0 current into the membrane.
            Only measures short traces to reduce time needed
        """
        self.pre_measure(neuron, current=None)
        old_recording_time = self.sthal.recording_time
        self.sthal.adc.setRecordingTime(1e-4)
        times, trace = self.sthal.read_adc()
        trace = self.readout_shifts(neuron, trace)
        V_rest = np.mean(trace)
        std    = np.std(trace)
        self.logger.TRACE("Measured V_rest of neuron {0}: {1:.3f} V".format(neuron, V_rest))
        self.sthal.adc.setRecordingTime(old_recording_time)
        return V_rest, std

    def find_best_current(self, neuron, V_rest, currents, threshold=0.15, recording_time_divider=20., skip_I_gl=700):
        """ Sweep over a range of currents and find the best current for fit.

            Args:
                neuron: coordinate
                V_rest: float. where is V_rest?
                currents: currents to sweep
                threshold: how far should the trace go above V_rest?
                recording_time_divider: how much smaller should the recorded traces be? \
                        Divider 1 means ~60 cycles (5e-3 s), standard divider of 20 means 3 cycles (2.5e-4 s)
                skip_I_gl: Skip the measurement above a certain I_gl and just use 30 nA.
        """
        old_recording_time = self.sthal.recording_time
        I_gl = self.get_parameter(neuron_parameter.I_gl, neuron).values()[0]*2500/1023.
        if I_gl > skip_I_gl: # Experience shows that for a large enough I_gl, a current of 30 nA is good
            self.logger.TRACE("I_gl of {0:.1f} large enough. Setting current to 30 nA".format(I_gl))
            return 30
        self.sthal.adc.setRecordingTime(old_recording_time/recording_time_divider)
        self.logger.TRACE("Finding best current for neuron {}".format(neuron))
        best_current = None
        highest_trace_max = None
        for current in currents:
            self.pre_measure(neuron, current=current)
            times, trace = self.sthal.read_adc()
            trace = self.readout_shifts(neuron, trace)
            trace_max = np.max(trace)
            if (trace_max - V_rest) < threshold: # Don't go higher than 150 mV above V_rest
                highest_trace_max = trace_max
                best_current = current
            else:
                break
        self.sthal.adc.setRecordingTime(old_recording_time)
        if best_current is None and highest_trace_max is None:
            self.logger.WARN("No best current found for neuron {}. Using 1 nA".format(neuron))
            return 1
        else:
            return best_current

    def _measure(self, analyzer):
        """ Measure traces and correct each value for readout shift.
            Also applies analyzer to measurement
        """
        self.logger.INFO("Measuring.")
        worker = WorkerPool(analyzer)
        for neuron in self.neurons:
            V_rest, std = self.measure_V_rest(neuron)
            current = self.find_best_current(neuron, V_rest, currents=self.currents, skip_I_gl=self.skip_I_gl)
            self.logger.TRACE("Measuring neuron {} with current {}".format(neuron, current))
            self.pre_measure(neuron, current=current)
            times, trace = self.sthal.read_adc()
            worker.do(neuron, times, self.readout_shifts(neuron, trace), neuron, std, V_rest, current)
            if not self.traces is None:
                self.traces[neuron] = np.array([times, trace])
        self.logger.INFO("Wait for analysis to complete.")
        return worker.join()

    def pre_measure(self, neuron, current, stim_length=65):
        if current is None:
            self.sthal.switch_analog_output(neuron, l1address=None) # No firing activated
        else:
            self.sthal.set_current_stimulus(int(current), stim_length)
            self.sthal.switch_current_stimulus_and_output(neuron, l1address=None)


class I_gl_Measurement_multiple_currents(I_gl_Measurement):
    def __init__(self, sthal, neurons, readout_shifts=None, currents=[10,35,70]):
        super(I_gl_Measurement, self).__init__(sthal, neurons, readout_shifts)
        self.currents = currents

    def _measure(self, analyzer):
        """ Measure traces and correct each value for readout shift.
            Also applies analyzer to measurement
        """
        self.logger.INFO("Measuring.")
        worker = WorkerPool(analyzer)
        for neuron in self.neurons:
            V_rest, std = self.measure_V_rest(neuron)
            self.logger.INFO("Measuring neuron {0} with currents {1}. V_rest = {2:.2f}+-{3:.2f}".format(neuron,
                                                                                    self.currents, V_rest, std))
            if not self.traces is None:
                self.traces[neuron] = []
            for current in self.currents:
                self.logger.TRACE("Measuring neuron {} with current {}".format(neuron, current))
                self.pre_measure(neuron, current)
                times, trace = self.sthal.read_adc()
                worker.do((neuron, current), times, self.readout_shifts(neuron, trace), neuron, std, V_rest)
                if not self.traces is None:
                    self.traces[neuron].append((current, np.array([times, trace])))
        self.logger.INFO("Wait for analysis to complete.")
        return worker.join()
