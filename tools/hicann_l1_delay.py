#! /usr/bin/python
# -*- coding: utf-8 -*-

import copy
import os

import Coordinate
import pyhalbe
import pysthal
import pylogging
import numpy

from collections import defaultdict

from pysthal.command_line_util import add_default_coordinate_options
from pysthal.command_line_util import add_logger_options
from pysthal.command_line_util import init_logger

from pycake.helpers.units import DAC, Voltage, Current
from pycake.helpers.calibtic import Calibtic

from pyhalbe.HICANN import neuron_parameter as np

from Coordinate import X, Y, Enum, top, bottom, left, right
from Coordinate import DNCMergerOnHICANN
from Coordinate import GbitLinkOnHICANN
from Coordinate import NeuronOnHICANN
from Coordinate import SynapseDriverOnHICANN
from Coordinate import RowOnSynapseDriver


REPEATER = {
    np.E_l: Voltage(700, apply_calibration=True),
    np.V_t: Voltage(727, apply_calibration=True),
    np.E_synx: Voltage(900, apply_calibration=True),
    np.E_syni: Voltage(500, apply_calibration=True),
    np.V_syntcx: Voltage(0, apply_calibration=True),
    np.V_syntci: Voltage(0, apply_calibration=True),
    np.I_gl: Current(0, apply_calibration=True),
    np.I_pl: Current(2500, apply_calibration=False),
}

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
    NO_N = 512
    PER_RUN = 16
    STEPS = NO_N/PER_RUN
    NEURON_BLOCKS = [
        [NeuronOnHICANN(Enum(ii)) for ii in range(start, start + PER_RUN)]
            for start in range(0, NO_N, PER_RUN)]

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

    def __init__(self, calibration_path):

        self.analog = Coordinate.AnalogOnHICANN(0)
        self.bg_l1address = pyhalbe.HICANN.L1Address(0)
        self.stim_l1address = pyhalbe.HICANN.L1Address(16)
        self.weight = pyhalbe.HICANN.SynapseWeight(15)
        self.gmax_div = 11
        self.driver_row = RowOnSynapseDriver(0)
        self.calibration_path = calibration_path
        self.stim_spike_times = numpy.arange(100) * 2.0e-5 + 2.0e-5
        self.runtime = self.stim_spike_times[-1] + 2.0e-5

    def run(self, wafer_coordinate, hicann_coordinate):
        wafer = pysthal.Wafer(wafer_coordinate)
        hicann = wafer[hicann_coordinate]

        self.initial_configuration(wafer, hicann)
        wafer.connect(pysthal.MagicHardwareDatabase())
        wafer.configure(pysthal.HICANNConfigurator())

        update_cfg = self.CFG(hicann.synapses)
        runner = pysthal.ExperimentRunner(self.runtime)
        neuron_spikes = defaultdict(list)
        stim_spikes = {}
        for step in xrange(self.STEPS):
            neurons = self.NEURON_BLOCKS[step]
            link_out, link_stim_out, neuron_addresses = self.configure_step(
                    wafer, hicann, neurons, False)
            wafer.configure(update_cfg)
            # adc = hicann.analogRecorder(self.analog)
            # adc.activateTrigger(self.runtime)
            wafer.start(runner)
            # trace = adc.trace()
            # adc.freeHandle()
            # numpy.save('membrane_{}.dat'.format(step), trace)

            rm_neurons = set(neuron_addresses[int(addr)]
                for addr in hicann.receivedSpikes(link_out)[:,1] if int(addr) in neuron_addresses)
            neurons = sorted(set(neurons) - rm_neurons)

            link_out, link_stim_out, neuron_addresses = self.configure_step(
                    wafer, hicann, neurons, True)
            wafer.configure(update_cfg)
            wafer.start(runner)
            for t, addr in hicann.receivedSpikes(link_out):
                neuron_spikes[neuron_addresses.get(int(addr), None)].append(t)

            tmp = hicann.receivedSpikes(link_stim_out)
            tmp = tmp[tmp[:,1] == int(self.stim_l1address)][:,0]
            for neuron in neurons:
                stim_spikes[neuron] = tmp

        good = []
        for neuron in Coordinate.iter_all(NeuronOnHICANN):
            spikes = len(neuron_spikes[neuron])
            if neuron in stim_spikes and spikes == len(stim_spikes[neuron]):
                _, link_out = self.get_gbit_links(neuron)
                print "MATCH!!!", neuron
                dt = neuron_spikes[neuron] - stim_spikes[neuron]
                print "Delay: {} +- {} for {} merger".format(
                    numpy.mean(dt), numpy.std(dt), MTREE_DEPTH[link_out])

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

        hicann.synapses.set_all(self.stim_l1address.getSynapseDecoderMask(),
                                pyhalbe.HICANN.SynapseWeight(0));

    def configure_step(self, wafer, hicann, neurons, enable_stimulus):
        neuron0 = neurons[0]

        wafer.clearSpikes()
        hicann.disable_l1_output()
        hicann.clear_l1_routing()

        hicann.enable_aout(neuron0, self.analog)

        for merger in Coordinate.iter_all(Coordinate.DNCMergerOnHICANN):
            m = hicann.layer1[merger]
            m.loopback = False

        for channel in Coordinate.iter_all(GbitLinkOnHICANN):
            hicann.layer1[channel] = pyhalbe.HICANN.GbitLink.OFF

        link_in, link_out = self.get_gbit_links(neuron0)
        link_stim_out = GbitLinkOnHICANN(
                int(link_in) / 2 * 2 + (int(link_in) + 1) % 2)

        driver = self.get_synapse_driver(neuron0, link_in)

        hicann.layer1[link_in] = pyhalbe.HICANN.GbitLink.TO_HICANN
        hicann.layer1[link_out] = pyhalbe.HICANN.GbitLink.TO_DNC
        hicann.layer1[link_stim_out] = pyhalbe.HICANN.GbitLink.TO_DNC
        hicann.route(link_in.toOutputBufferOnHICANN(), driver)
        self.config_driver(hicann, driver, enable_stimulus)

        hicann.layer1[DNCMergerOnHICANN(int(link_in))].loopback = True

        synrow = Coordinate.SynapseRowOnHICANN(driver, self.driver_row)
        weights = hicann.synapses[synrow].weights

        neuron_addresses = {}
        for addr, neuron in enumerate(neurons, 64 - self.PER_RUN):
            addr = pyhalbe.HICANN.L1Address(addr)
            hicann.enable_l1_output(neuron, addr)
            neuron_addresses[int(addr)] = neuron
            weights[int(neuron.x())] = self.weight

        in_spikes = pysthal.Vector_Spike([
            pysthal.Spike(self.stim_l1address, t) for t in self.stim_spike_times])
        hicann.sendSpikes(link_in, in_spikes)

        return link_out, link_stim_out, neuron_addresses

    def get_synapse_driver(self, neuron, link):
        if neuron.y() == top:
            return SynapseDriverOnHICANN(Enum(97 + int(link) * 2))
        else:
            return SynapseDriverOnHICANN(Enum(126 - int(link) * 2))

    def get_gbit_links(self, neuron):
        link_out = GbitLinkOnHICANN(int(neuron.x()) / 32)
        if link_out == GbitLinkOnHICANN(7) or link_out == GbitLinkOnHICANN(6):
            link_in = GbitLinkOnHICANN(0)
        else:
            link_in = GbitLinkOnHICANN(7)
        return link_in, link_out

    def config_driver(self, hicann, driver, enable_stimulus):
        driver_decoder = self.stim_l1address.getDriverDecoderMask()
        drv = hicann.synapses[driver]
        drv[self.driver_row].set_decoder(top, driver_decoder)
        drv[self.driver_row].set_decoder(bottom, driver_decoder)
        drv[self.driver_row].set_gmax_div(left, self.gmax_div)
        drv[self.driver_row].set_gmax_div(right, self.gmax_div)
        drv[self.driver_row].set_syn_in(left, 1)
        drv[self.driver_row].set_syn_in(right, 0)
        drv[self.driver_row].set_gmax(0)
        if enable_stimulus:
            drv.set_l1()

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
        return


def main():
    import argparse

    def folder(value):
        if not os.path.isdir(value):
            try:
                os.makedirs(value)
            except Exception as err:
                msg = "Couldn't created output folder {}: {}".format(value, err)
                raise argparse.ArgumentTypeError(msg)
        if not os.access(value, os.R_OK | os.W_OK | os.X_OK):
            raise argparse.ArgumentTypeError(
                "{0} is not accessible".format(value))
        return value

    init_logger('ERROR')

    parser = argparse.ArgumentParser(
            description='Configure HICANN to stimulate a neuron via a second neuron')
    add_default_coordinate_options(parser)
    add_logger_options(parser)
    parser.add_argument('--calibration', action='store', type=folder,
            help="Path to calibration data")
    args = parser.parse_args()

    cmd = FindRepeaterNeurons(args.calibration)
    cmd.run(args.wafer, args.hicann)

if __name__ == '__main__':
    main()