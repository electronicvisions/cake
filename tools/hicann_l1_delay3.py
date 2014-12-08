"""
Measure HICANN L1 delay interactivly with scope

Result see https://brainscales-r.kip.uni-heidelberg.de/issues/1185
"""

import copy
import os
import cPickle

import Coordinate
import pyhalbe
import pysthal
import pylogging
import numpy

from collections import defaultdict

from pysthal.command_line_util import add_default_coordinate_options
from pysthal.command_line_util import add_logger_options
from pysthal.command_line_util import init_logger
from pysthal.command_line_util import folder

from pycake.helpers.units import DAC, Voltage, Current
from pycake.helpers.calibtic import Calibtic

from pyhalbe.HICANN import neuron_parameter as np
from pyhalbe.HICANN import shared_parameter as sp

from Coordinate import X, Y, Enum, top, bottom, left, right, iter_all
from Coordinate import DNCMergerOnHICANN
from Coordinate import FGBlockOnHICANN
from Coordinate import GbitLinkOnHICANN
from Coordinate import NeuronOnHICANN
from Coordinate import SynapseDriverOnHICANN
from Coordinate import RowOnSynapseDriver


REPEATER = {
    np.E_l: Voltage(900, apply_calibration=True),
    np.V_t: Voltage(850, apply_calibration=True),
    np.E_synx: Voltage(900, apply_calibration=True),
    np.E_syni: Voltage(500, apply_calibration=True),
    np.V_syntcx: Voltage(0, apply_calibration=True),
    np.V_syntci: Voltage(0, apply_calibration=True),
    np.I_gl: Current(0, apply_calibration=True),
    np.I_pl: Current(2200, apply_calibration=False),
}

VRESET = Voltage(500, apply_calibration=True)

MTREE_DEPTH = {
    GbitLinkOnHICANN(0) : 2,
    GbitLinkOnHICANN(1) : 3,
    GbitLinkOnHICANN(2) : 2,
    GbitLinkOnHICANN(3) : 5,
    GbitLinkOnHICANN(4) : 2,
    GbitLinkOnHICANN(5) : 4,
    GbitLinkOnHICANN(6) : 3,
    GbitLinkOnHICANN(7) : 2,
}

VSYNTCX_CORR = -12

logger = pylogging.get(__name__.replace('_', ''))

class FindRepeaterNeurons(object):
    class CFG(pysthal.HICANNConfigurator):
        def __init__(self, synapse_array):
            super(FindRepeaterNeurons.CFG, self).__init__()
            self.last_array = copy.copy(synapse_array)

        def config_fpga(self, handle, fpga):
            pass

        def config_synapse_array(self, handle, data):
            """ Differential update of synapse array """
            synapses = data.synapses
            for syndrv in Coordinate.iter_all(Coordinate.SynapseDriverOnHICANN):
                row = synapses.getDecoderDoubleRow(syndrv)
                if row != self.last_array.getDecoderDoubleRow(syndrv):
                    self.last_array.setDecoderDoubleRow(syndrv, row)
                    pyhalbe.HICANN.set_decoder_double_row(handle, syndrv, row)
                    self.getLogger().info("update decoder row {}".format(syndrv))

                for side in Coordinate.iter_all(Coordinate.SideVertical):
                    synrow = Coordinate.SynapseRowOnHICANN(syndrv, side)
                    row_data = synapses[synrow].weights
                    if row_data != self.last_array[synrow].weights:
                        self.last_array[synrow].weights = row_data
                        pyhalbe.HICANN.set_weights_row(handle, synrow, row_data)
                        self.getLogger().info("update synapse row {}".format(synrow))

        def hicann_init(self, handle):
            pass

        def config_floating_gates(self, handle, data):
            pass

        def config_fg_stimulus(self, handle, data):
            pass

        def config_phase(self, handle, data):
            pass

    class UpdateAoutCFG(pysthal.HICANNConfigurator):
        def config_fpga(self, *args):
            """do not reset FPGA"""
            pass

        def config(self, fpga_handle, h, hicann):
            """Call analog output related configuration functions."""
            # self.config_neuron_config(h, hicann);
            # self.config_neuron_quads(h, hicann)
            self.config_analog_readout(h, hicann)
            # self.config_fg_stimulus(h, hicann)
            self.flush_fpga(fpga_handle)

    def __init__(self, calibration_path):

        self.analog = Coordinate.AnalogOnHICANN(1)
        self.analog_preout = Coordinate.AnalogOnHICANN(0)
        self.bg_l1address = pyhalbe.HICANN.L1Address(0)
        self.neuron_l1address = pyhalbe.HICANN.L1Address(16)
        self.weight = pyhalbe.HICANN.SynapseWeight(15)
        self.gmax_div = 11
        self.driver_row = RowOnSynapseDriver(0)
        self.calibration_path = calibration_path
        self.runtime = 0.04

    def run(self, wafer_coordinate, hicann_coordinate):
        wafer = pysthal.Wafer(wafer_coordinate)
        hicann = wafer[hicann_coordinate]

        self.initial_configuration(wafer, hicann)
        wafer.connect(pysthal.MagicHardwareDatabase())
        wafer.configure(pysthal.HICANNConfigurator())

        runner = pysthal.ExperimentRunner(self.runtime)
        update_cfg = self.CFG(hicann.synapses)
        while True:
            try:
                raw = raw_input("Switch to neuron: ")
                neuron = NeuronOnHICANN(Enum(int(raw)))
            except EOFError:
                break
            except (ValueError, RuntimeError) as err:
                print "Invalid input '{}', {}.".format(raw, err)
                continue

            link_out = self.configure_step(wafer, hicann, neuron)
            hicann.analog.set_preout(self.analog_preout)
            wafer.configure(update_cfg)

    def initial_configuration(self, wafer, hicann):
        for merger in Coordinate.iter_all(DNCMergerOnHICANN):
            m = hicann.layer1[merger]
            m.config = m.MERGE
            m.slow = True
            m.loopback = False

        for bg in Coordinate.iter_all(Coordinate.BackgroundGeneratorOnHICANN):
            hicann.layer1[bg].enable(True)
            hicann.layer1[bg].random(False)
            hicann.layer1[bg].period(3000)
            hicann.layer1[bg].address(self.bg_l1address)

        for channel in Coordinate.iter_all(GbitLinkOnHICANN):
            hicann.layer1[channel] = pyhalbe.HICANN.GbitLink.OFF

        self.set_neuron_parameters(wafer, hicann)

        hicann.synapses.set_all(self.neuron_l1address.getSynapseDecoderMask(),
                                pyhalbe.HICANN.SynapseWeight(0));

        in_spikes = pysthal.Vector_Spike([
            pysthal.Spike(self.bg_l1address, self.runtime)])
        hicann.sendSpikes(GbitLinkOnHICANN(7), in_spikes)

    def configure_step(self, wafer, hicann, neuron):
        hicann.disable_l1_output()
        hicann.clear_l1_routing()

        hicann.enable_aout(neuron, self.analog)

        link_out = self.get_gbit_link(neuron)
        driver = self.get_synapse_driver(neuron, link_out)
        hicann.layer1[link_out] = pyhalbe.HICANN.GbitLink.TO_DNC
        hicann.route(link_out.toOutputBufferOnHICANN(), driver)
        self.config_driver(hicann, driver)

        synrow = Coordinate.SynapseRowOnHICANN(driver, self.driver_row)
        weights = hicann.synapses[synrow].weights

        hicann.enable_l1_output(neuron, self.neuron_l1address)
        weights[int(neuron.x())] = self.weight

        return link_out


    def get_synapse_driver(self, neuron, link):
        if neuron.y() == top:
            return SynapseDriverOnHICANN(Enum(97 + int(link) * 2))
        else:
            return SynapseDriverOnHICANN(Enum(126 - int(link) * 2))

    def get_gbit_link(self, neuron):
        return GbitLinkOnHICANN(int(neuron.x()) / 32)

    def config_driver(self, hicann, driver):
        driver_decoder = self.neuron_l1address.getDriverDecoderMask()
        drv = pyhalbe.HICANN.SynapseDriver()
        drv[top].set_decoder(top, driver_decoder)
        drv[top].set_decoder(bottom, driver_decoder)
        drv[bottom].set_decoder(top, driver_decoder)
        drv[bottom].set_decoder(bottom, driver_decoder)
        drv[self.driver_row].set_gmax_div(left, self.gmax_div)
        drv[self.driver_row].set_gmax_div(right, self.gmax_div)
        drv[self.driver_row].set_syn_in(left, 0)
        drv[self.driver_row].set_syn_in(right, 1)
        drv[self.driver_row].set_gmax(0)
        drv.set_l1()
        hicann.synapses[driver] = drv

    def set_neuron_parameters(self, wafer, hicann):
        fgc = hicann.floating_gates

        # load calibration data
        calib = Calibtic(self.calibration_path, wafer.index(), hicann.index())

        for neuron in Coordinate.iter_all(NeuronOnHICANN):
            for param, value in REPEATER.iteritems():
                value_dac = value.toDAC()
                if value_dac.apply_calibration:
                    value_dac.value = calib.apply_calibration(
                      value_dac.value, param, neuron)
                if param == np.V_syntcx:
                    value_dac.value += VSYNTCX_CORR
                fgc.setNeuron(neuron, param, value_dac.value)
        for block in Coordinate.iter_all(FGBlockOnHICANN):
            param = sp.V_reset
            value_dac = VRESET.toDAC()
            if value_dac.apply_calibration:
                value_dac.value = calib.apply_calibration(
                    value_dac.value, param, block)
            fgc.setShared(block, param, value_dac.value)

def main():
    import argparse
    init_logger('INFO')

    parser = argparse.ArgumentParser(
            description='Configure HICANN to stimulate a neuron via a second neuron')
    add_default_coordinate_options(parser)
    add_logger_options(parser)
    parser.add_argument('--calibration', action='store', type=folder,
            help="Path to calibration data")
    args = parser.parse_args()

    N = [n for n in iter_all(NeuronOnHICANN)]
    cmd = FindRepeaterNeurons(args.calibration)
    #cmd.run(args.wafer, args.hicann, N[:10] + N[96:108])
    cmd.run(args.wafer, args.hicann)

if __name__ == '__main__':
    main()
