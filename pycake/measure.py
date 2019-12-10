import numpy as np
import pylogging
from Coordinate import NeuronOnHICANN, FGBlockOnHICANN, iter_all, GbitLinkOnHICANN
import Coordinate
from pyhalbe.HICANN import neuron_parameter, shared_parameter
import pyhalbe
import pysthal
import pycake

from helpers.WorkerPool import FakeWorkerPool, WorkerPool, DEFAULT_WORKERS
from helpers.TracesOnDiskDict import PandasRecordsOnDiskDict
from helpers.sthal import ADCFreqConfigurator
from tables.exceptions import HDF5ExtError

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
    """ Base class for the measurement protocol

        Args:
            sthal: fully configured StHALContainer as found in pycake.helpers.sthal
            neurons: list of NeuronOnHICANN coordinates
    """

    logger = pylogging.get("pycake.measurement")

    def __init__(self, sthal, neurons, readout_shifts=None, workers=DEFAULT_WORKERS):
        for neuron in neurons:
            if not isinstance(neuron, NeuronOnHICANN):
                raise TypeError("Expected list of integers")
        # TODO: callable readout_shifter instead of dict
        # readout_shifter(neuron, trace)
        self.sthal = sthal
        self.step_parameters = {}
        self.neurons = neurons
        self.time_created = time.asctime()
        self.run_time = -1.0
        self.traces = None
        self.done = False
        self.workers = workers

    def save_traces(self, storage_path):
        """Enabling saving of traces to the given file"""
        self.traces = PandasRecordsOnDiskDict(*os.path.split(storage_path))
        try:
            # open store to see if it is corrupt/not existing
            self.traces.h5file
        except HDF5ExtError:
            # store could not be opened -> corrupt -> remove it
            # (this is possible as each measurement is stored in its own store)
            self.logger.warn("Trace store is corrupt. Store will be removed. "
                             "Storing traces at '{}'".format(storage_path))
            os.remove(storage_path)

    def finish(self):
        """ Finish the measurement.
            This sets the 'done' flag and time_finished.
        """
        self.done = True
        self.time_finished = time.asctime()
        if self.traces is not None:
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
                parameter: [list] parameters to read (neuron_parameter,
                           shared_parameter or string). Floating gate
                           parameters will be read from the floating gate
                           configuration. Sting parameters will be returned
                           from step_parameters.
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
                try:
                    values.append(fgs.getShared(block, parameter))
                except IndexError:
                    values.append(np.nan)
            elif parameter is None:
                continue
            elif isinstance(parameter, basestring):
                try:
                    values.append(self.step_parameters[parameter])
                except KeyError:
                    values.append(np.nan)
            else:
                raise TypeError("invalid parameter type {}".format(type(parameter)))
        return values

    def get_neurons(self):
        return self.neurons

    def configure(self, configurator):
        """ Write StHALContainer configuration to the hardware.
            This connects to the hardware.
        """
        self.sthal.write_config(configurator=configurator)

    def pre_measure(self, neuron, l1address=None):
        """Hook to be execute before a measurement is taken"""
        pass

    def get_readout_hicann(self):

        readout_wafer = pysthal.Wafer()
        HRC = pysthal.HICANNReadoutConfigurator(readout_wafer)
        self.sthal.wafer.configure(HRC)
        return readout_wafer[self.sthal.coord_hicann]

    def _measure(self, analyzer, additional_data):
        """ Measure traces and correct each value for readout shift.
            Changes traces to numpy arrays

            This will set:
                self.traces = {neuron: trace}
        """
        raise NotImplementedError("Not implemented in {}".format(
            type(self).__name__))

    def run_measurement(self, analyzer, additional_data,
                        configurator=None, disconnect=True):
        """First configure, then measure"""
        t_start = time.time()
        self.logger.INFO("Connecting to hardware and configuring.")
        self.configure(configurator)
        result = self._measure(analyzer, additional_data)
        self.finish()
        if disconnect:
            self.sthal.disconnect()
        self.tun_time = time.time() - t_start
        self.logger.INFO("Measurement done.")
        return result

    def get_worker_pool(self, analyzer):
        """Creates a worker pool with the given analyzer"""
        if self.workers is None:
            return FakeWorkerPool(analyzer)
        else:
            return WorkerPool(analyzer, self.workers)


class SpikeMeasurement(Measurement):

    def __init__(self, sthal, neurons, readout_shifts=None, workers=DEFAULT_WORKERS):
        super(SpikeMeasurement, self).__init__(sthal, neurons, workers=workers)

        self.spikes = {}

    def get_spikes(self, nbs):
        """

        nbs: neuron blocks to be enabled

        TODO: this would better fit into the StHALContainer, like read_adc
        """

        self.logger.info("Enabling L1 output for neurons on NeuronBlock {}".format(nbs))

        self.sthal.wafer.clearSpikes()

        self.sthal.hicann.disable_aout()
        self.sthal.hicann.disable_l1_output()

        activated_neurons = []

        for n, neuron in enumerate(self.neurons):

            for i in nbs:

                if neuron.toEnum().value() >= 32*i and neuron.toEnum().value() < 32*(i+1):
                    addr = neuron.toNeuronOnNeuronBlock().x().value() + neuron.toNeuronOnNeuronBlock().y().value()*32

                    self.sthal.hicann.enable_l1_output(neuron, pyhalbe.HICANN.L1Address(addr))
                    self.sthal.hicann.neurons[neuron].activate_firing(True)

                    activated_neurons.append(neuron)

        for channel in Coordinate.iter_all(Coordinate.GbitLinkOnHICANN):
            self.sthal.hicann.layer1[channel] = pyhalbe.HICANN.GbitLink.Direction.TO_DNC

        self.sthal.wafer.configure(pycake.helpers.sthal.SpikeReadoutHICANNConfigurator())

        runner = pysthal.ExperimentRunner(0.1e-3)
        self.sthal.wafer.start(runner)

        no_spikes = []
        spiketimes = []
        spikeaddresses = []

        addr_spikes = defaultdict(list)

        for ii, channel in enumerate(iter_all(GbitLinkOnHICANN)):
            received = self.sthal.hicann.receivedSpikes(channel)
            times, addresses = received.T

            self.logger.info("received {} spike(s) on channel {}".format(len(received), channel))

            no_spikes.append(len(received))
            spiketimes.append(times)
            spikeaddresses.append(addresses)

            for t, a in zip(times, addresses):
                if a < 32:
                    n = a + ii*32
                else:
                    n = 256 + (a-32) + ii*32

                addr_spikes[Coordinate.NeuronOnHICANN(Coordinate.Enum(int(n)))].append(t)

        for neuron, spikes in addr_spikes.iteritems():

            if neuron in activated_neurons:
                self.spikes[neuron] = spikes

            if neuron not in activated_neurons and len(spikes) != 0:
                self.logger.warn("Neuron {} was not activated, but spiked {} time(s)".format(neuron, len(spikes)))

        self.sthal.hicann.disable_aout()
        self.sthal.hicann.disable_l1_output()

    def _measure(self, analyzer, additional_data):
        """ Spikes
            Changes traces to numpy arrays
        """
        self.adc_status = []
        self.logger.INFO("Measuring.")

        results = {}
        self.logger.INFO("Reading out spikes")

        for nb in xrange(0, 16):
            self.get_spikes([nb])

        for neuron in self.neurons:
            spikes = self.spikes.get(neuron, [])
            results[neuron] = analyzer(neuron, spikes)

        return results


class ADCMeasurement(Measurement):
    """ This implements the Measurement protocol for an ADC

        Args:
            sthal: fully configured StHALContainer as found in pycake.helpers.sthal
            neurons: list of NeuronOnHICANN coordinates
            readout_shifts (optional): Dictionary {neuron: shift} in V that contains
                the individual readout shifts extracted from V_reset measurements.
                Note: Every voltage value from the trace is shifted back by -shift!
                If None is given, the shifts are set to 0.
    """
    def __init__(self, sthal, neurons, readout_shifts=None, workers=DEFAULT_WORKERS):
        super(ADCMeasurement, self).__init__(sthal, neurons, workers=workers)

        # Debug for repeated ADC traces
        self.last_trace = None
        self.adc_status = []

        if readout_shifts is None:
            self.logger.WARN("No readout shifts found. Shifts are set to 0")
            self.readout_shifts = no_shift
        else:
            self.readout_shifts = ReadoutShift(readout_shifts)

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

        data = self.traces[neuron]
        assert len(data.index.values) == len(data['v'].values)
        if apply_readout_shift:
            data['v'] = self.readout_shifts(neuron, data['v'])
        assert len(data.index.values) == len(data['v'].values)
        return data.index.values, data['v'].values

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

    def read_adc(self):
        """Read ADC. Override for spikes.

        Returns:
            tuple: (traces, None)
        """
        return self.sthal.read_adc(), None

    def pre_measure(self, neuron, l1address=None):
        """Hook to be execute before a measurement is taken"""
        self.sthal.switch_analog_output(neuron, l1address=l1address)

    def _measure(self, analyzer, additional_data):
        """ Measure traces and correct each value for readout shift.
            Changes traces to numpy arrays

            This will set:
                self.traces = {neuron: trace}
        """
        self.last_trace = np.array([])
        self.adc_status = []
        # self.logger.INFO("Measure FG values.")
        # self.fg_values = self.sthal.read_floating_gates()
        self.logger.INFO("Measuring.")

        self.logger.INFO("Reading out traces")
        with self.get_worker_pool(analyzer) as worker:
            for neuron in self.neurons:
                self.pre_measure(neuron)
                readout, spikes = self.read_adc()
                if self.traces is not None:
                    self.traces[neuron] = readout
                readout['v'] = self.readout_shifts(neuron, readout['v'])
                worker.do(neuron, neuron=neuron, trace=readout,
                    spikes=spikes, additional_data=additional_data)

                # DEBUG stuff
                self.adc_status.append(self.sthal.read_adc_status())
                if np.array_equal(readout['v'], self.last_trace):
                    self.logger.ERROR(
                        "ADC trace didn't change from the last "
                        "readout, printing status information of all ADC "
                        "previous readouts:\n" + "\n".join(self.adc_status))
                    raise RuntimeError(
                        "Broken ADC readout abort measurement (details see "
                        "log messages)")
                self.last_trace = readout['v']
                # DEBUG stuff end

            self.last_trace = None
            self.logger.INFO("Wait for analysis to complete.")
            return worker.join()


class ADCMeasurementWithSpikes(ADCMeasurement):
    def read_adc(self):
        """Read ADC. Override for spikes.

        Returns:
            tuple: (traces, spikes)
        """
        return self.sthal.read_adc_and_spikes()


class I_gl_Measurement(ADCMeasurement):
    """ This measurement is done in three steps:

        1) Measure V_rest without injecting current
        2) Vary input current between 1 to 30 to determine which one generates the best traces
        3) Measure the main trace that is used for the fit

        This takes a lot of time!

    """
    def __init__(self, sthal, neurons, readout_shifts=None, currents=[5, 10, 15, 20, 25, 30], workers=DEFAULT_WORKERS):
        super(I_gl_Measurement, self).__init__(sthal, neurons, readout_shifts, workers)
        self.currents = currents

    def measure_V_rest(self, neuron):
        """ Measure V_rest and independend std by injecting 0 current into the membrane.
            Only measures short traces to reduce time needed
        """
        self.pre_measure(neuron, current=None)
        old_recording_time = self.sthal.recording_time
        self.sthal.adc.setRecordingTime(1e-4)
        readout, _ = self.read_adc()
        trace = self.readout_shifts(neuron, readout['v'])
        V_rest = np.mean(trace)
        std = np.std(trace)
        self.logger.TRACE("Measured V_rest of neuron {0}: {1:.3f} V".format(neuron, V_rest))
        self.sthal.adc.setRecordingTime(old_recording_time)
        return V_rest, std

    def find_best_current(self, neuron, V_rest, currents, threshold=0.15, recording_time_divider=20.):
        """ Sweep over a range of currents and find the best current for fit.

            Args:
                neuron: coordinate
                V_rest: float. where is V_rest?
                currents: currents to sweep
                threshold: how far should the trace go above V_rest?
                recording_time_divider: how much smaller should the recorded traces be? \
                        Divider 1 means ~60 cycles (5e-3 s), standard divider of 20 means 3 cycles (2.5e-4 s)
        """
        old_recording_time = self.sthal.recording_time
        self.sthal.adc.setRecordingTime(old_recording_time/recording_time_divider)
        self.logger.TRACE("Finding best current for neuron {}".format(neuron))
        best_current = None
        highest_trace_max = None
        for current in currents:
            self.pre_measure(neuron, current=current)
            readout, _ = self.read_adc()
            trace = self.readout_shifts(neuron, readout['v'])
            trace_max = np.max(trace)
            trace_min = np.min(trace)
            if (trace_min - V_rest) > threshold:
                # reset has happened, neuron spiked,
                # stimulus is too strong
                self.logger.WARN("Neuron {} spiked at stimulus {}".format(neuron, current))
                break
            elif np.std(trace) < 0.008:
                # response may be too small
                self.logger.WARN("Neuron {} shows almost no response at stimulus {}".format(neuron, current))
                continue
            if (trace_max - V_rest) < threshold:
                # Don't go higher than 150 mV above V_rest
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

    def _measure(self, analyzer, additional_data):
        """ Measure traces and correct each value for readout shift.
            Also applies analyzer to measurement
        """
        self.logger.INFO("Measuring.")
        worker = self.get_worker_pool(analyzer)
        for neuron in self.neurons:
            V_rest, std = self.measure_V_rest(neuron)
            current = self.find_best_current(neuron, V_rest, currents=self.currents)
            self.logger.TRACE("Measuring neuron {} with current {}".format(neuron, current))
            self.pre_measure(neuron, current=current)
            readout, _ = self.read_adc()

            # these are just an integers/floats, convert to array for TracesOnDiskDict
            tmp = additional_data.copy()
            tmp['current'] = np.array([current])
            tmp['std'] = np.array([std])
            tmp['V_rest'] = np.array([V_rest])

            if self.traces is not None:
                self.traces[neuron] = readout

            readout['v'] = self.readout_shifts(neuron, readout['v'])
            worker.do(neuron, neuron=neuron, trace=readout,
                      additional_data=tmp)
        self.logger.INFO("Wait for analysis to complete.")
        return worker.join()

    def pre_measure(self, neuron, current, stim_length=65):
        if current is None:
            self.sthal.switch_analog_output(neuron, l1address=None)
        else:
            self.sthal.set_current_stimulus(int(current), stim_length)
            self.sthal.switch_current_stimulus_and_output(neuron, l1address=None)


class I_gl_Measurement_multiple_currents(I_gl_Measurement):
    """Does not find the best current, but instead measures once with each current."""
    def __init__(self, sthal, neurons, readout_shifts=None, currents=[10, 35, 70], workers=DEFAULT_WORKERS):
        super(I_gl_Measurement_multiple_currents, self).__init__(sthal, neurons, readout_shifts, workers)
        self.currents = currents

    def _measure(self, analyzer, additional_data):
        """ Measure traces and correct each value for readout shift.
            Also applies analyzer to measurement
        """
        self.logger.INFO("Measuring.")
        worker = self.get_worker_pool(analyzer)
        for neuron in self.neurons:
            V_rest, std = self.measure_V_rest(neuron)
            self.logger.INFO("Measuring neuron {0} with currents {1}. V_rest = {2:.2f}+-{3:.2f}".format(neuron, self.currents, V_rest, std))
            if self.traces is not None:
                self.traces[neuron] = {}
            for current in self.currents:
                self.logger.TRACE("Measuring neuron {} with current {}".format(neuron, current))
                self.pre_measure(neuron, current)
                readout, _ = self.read_adc()
                readout['current'] = current
                readout['std'] = std
                readout['V_rest'] = V_rest
                if self.traces is not None:
                    self.traces[neuron] = readout
                readout['v'] = self.readout_shifts(neuron, readout['v'])
                worker.do(neuron, neuron=neuron, trace=readout,
                          additional_data=additional_data)
        self.logger.INFO("Wait for analysis to complete.")
        return worker.join()

class ADCFreq_Measurement(ADCMeasurement):
    def __init__(self, sthal, neurons, bg_rate=100e3, readout_shifts=None):
        super(ADCFreq_Measurement, self).__init__(
            sthal, neurons, readout_shifts=readout_shifts)
        self.bg_rate = self.sthal.stimulatePreout(bg_rate)
        # Record about 2000 spikes
        self.sthal.set_recording_time(1.0/self.bg_rate, 2000.0)

    def _measure(self, analyzer, additional_data):
        """ Measure traces and correct each value for readout shift.
            Changes traces to numpy arrays

            This will set:
                self.traces = {neuron: trace}
        """
        time.sleep(0.2)  # Settle driver locking
        readout, _ = self.read_adc()
        # readout.to_hdf('dump.hdf', 'ADCFreq')
        if self.traces is not None:
            self.traces[NeuronOnHICANN()] = readout

        result = analyzer(trace=readout, bg_rate=self.bg_rate)
        self.logger.INFO("ADC Frequency measured: {}".format(result))
        return result

    def run_measurement(self, analyzer, additional_data,
                        configurator=None, disconnect=True):
        """ First configure, then measure
        """
        return ADCMeasurement.run_measurement(
             self, analyzer, additional_data, configurator=ADCFreqConfigurator())
