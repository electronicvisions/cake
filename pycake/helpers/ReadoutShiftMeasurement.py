import numpy as np
import time
import os
import pylogging

from pyhalbe.HICANN import neuron_parameter, shared_parameter
import pyhalbe

from pycake.helpers.sthal import StHALContainer
from pycake.measure import ADCMeasurement

import Coordinate as C

def nrn(nid):
    return C.NeuronOnHICANN(C.Enum(nid))
def blk(bid):
    return C.FGBlockOnHICANN(C.Enum(bid))

class ReadoutShiftMeasurement(ADCMeasurement):
    """
    """
    logger = pylogging.get("pycake.ReadoutShiftMeasurement")

    def __init__(self, base_parameters, wafer_coord, hicann_coord, E_l=0.9):

        self.neurons = [nrn(nid) for nid in range(512)]
        self.blocks = [blk(bid) for bid in range(4)]
        self.time_created = time.time()
        self.spikes = {}
        self.done = False
        self.traces = None
        self.last_trace = None
        self.adc_status = []

        self.E_l = E_l
        self.V_rests = {}

        self.neuron_size=64
        self.sthal = self.create_sthal(base_parameters, wafer_coord, hicann_coord, self.neuron_size)

    def prepare_parameters(self, sthal, parameters):
        """ Writes parameters into a sthal container.
        """

        fgc = pyhalbe.HICANN.FGControl()

        for neuron in self.neurons:
            for param, value in parameters.iteritems():
                if isinstance(param, shared_parameter) or param.name[0] == '_':
                    continue

                self.logger.TRACE("Setting FGValue of {} parameter {} to {}.".format(neuron, param, value))
                sthal.hicann.floating_gates.setNeuron(neuron, param, value.toDAC().value)

        for block in self.blocks:
            for param, value in parameters.iteritems():
                if isinstance(param, neuron_parameter) or param.name[0] == '_':
                    continue
                # Check if parameter exists for this block
                even = block.id().value()%2
                if even and param.name in ['V_clra', 'V_bout']:
                    continue
                if not even and param.name in ['V_clrc', 'V_bexp']:
                    continue

                self.logger.TRACE("Setting FGValue of {} parameter {} to {}.".format(block, param, value))
                sthal.hicann.floating_gates.setShared(block, param, value.toDAC().value)

        return sthal

    def create_sthal(self, base_parameters, wafer_coord, hicann_coord, neuron_size):
        """ Generates sthal container to measure readout shifts
        """
        sthal = StHALContainer(wafer_coord, hicann_coord, neuron_size=neuron_size)
        sthal = self.prepare_parameters(sthal, base_parameters)
        for neuron in self.neurons:
            sthal.hicann.floating_gates.setNeuron(neuron, neuron_parameter.I_convi, 0)
            sthal.hicann.floating_gates.setNeuron(neuron, neuron_parameter.I_convx, 0)
            sthal.hicann.floating_gates.setNeuron(neuron, neuron_parameter.I_gl,  800)
            sthal.hicann.floating_gates.setNeuron(neuron, neuron_parameter.E_l,  int(round(self.E_l*1000*1023/1800.)))
        return sthal

    def pre_measure(self, neuron):
        self.sthal.switch_analog_output(neuron)

    def _measure(self):
        """ Measure traces and correct each value for readout shift.
            Changes traces to numpy arrays

            This will set:
                self.traces = {neuron: trace}
        """
        self.last_trace = np.array([])
        self.adc_status = []
        self.logger.INFO("Measuring.")

        self.logger.INFO("Reading out traces")
        for neuron in self.neurons:
            self.pre_measure(neuron)
            readout = self.read_adc()
            if not self.traces is None:
                self.traces[neuron] = readout
            self.V_rests[neuron] = np.mean(readout['v'])

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
            self.last_readout = readout['v']
            # DEBUG stuff end
        self.last_trace = None

    def get_neuron_block(self, block_id, size):
        if block_id * size > 512:
            raise ValueError, "There are only {} blocks of size {}".format(512/size, size)
        nids_top = np.arange(size/2*block_id,size/2*block_id+size/2)
        nids_bottom = np.arange(256+size/2*block_id,256+size/2*block_id+size/2)
        nids = np.concatenate([nids_top, nids_bottom])
        neurons = [nrn(int(nid)) for nid in nids]
        return neurons

    def get_readout_shifts(self):
        """
        """
        n_blocks = 512/self.neuron_size
        readout_shifts = {}
        for block_id in range(n_blocks):
            neurons = self.get_neuron_block(block_id, self.neuron_size)
            V_rests = {neuron: self.V_rests[neuron] for neuron in neurons}
            mean_V_rest = np.mean(V_rests.values())
            for neuron in neurons:
                readout_shifts[neuron] = [float(self.V_rests[neuron] - mean_V_rest)]
        return readout_shifts

    def run_measurement(self):
        """ First configure, then measure
        """
        self.logger.INFO("Connecting to hardware and configuring.")
        self.configure(None)
        self._measure()
        self.logger.INFO("Measurement done, disconnecting from hardware.")
        self.finish()
        self.sthal.disconnect()
        self.logger.INFO("Calculating readout shifts from measurement.")
        self.readout_shifts = self.get_readout_shifts()
        return self.readout_shifts
