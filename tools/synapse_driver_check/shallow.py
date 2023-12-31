#! /usr/bin/python
# -*- coding: utf-8 -*-

import time
import numpy as np
import copy

from pyhalco_common import Enum, top, bottom, left, right, Y, iter_all
import pyhalco_hicann_v2 as Coordinate
import pyhalbe
from pyhalbe.HICANN import neuron_parameter
from pyhalbe.HICANN import shared_parameter
import pysthal
from pycake.helpers.calibtic import Calibtic
from pycake.helpers.units import DAC
from pycake.helpers.sthal import HICANNConfigurator

from utils import ValueStorage

try:
    import logbook
    log = logbook.Logger("shallow")
except:
    import logging
    log = logging.getLogger("shallow")

# Set up shortcuts
HICANN = pyhalbe.HICANN

# Select L1 addresses
BG_ADDRESS = pyhalbe.HICANN.L1Address(0)
TOP = top
BOT = bottom

class Parameters(ValueStorage):
    def __init__(self):
        ValueStorage.__init__(self, {
            'base_parameters': {
                shared_parameter.V_fac: DAC(1023),
                shared_parameter.V_dep: DAC(0),
                shared_parameter.V_stdf: DAC(0),
                shared_parameter.V_reset: DAC(284),
                shared_parameter.V_dtc: DAC(0),
                shared_parameter.V_gmax0: DAC(50),
                shared_parameter.V_gmax1: DAC(50),
                shared_parameter.V_gmax2: DAC(50),
                shared_parameter.V_gmax3: DAC(50),
                shared_parameter.V_bstdf: DAC(0),
                },
            'neuron_parameters': {
                nrn : {} for nrn in
                iter_all(Coordinate.NeuronOnHICANN)},
            'shared_parameters': {
                blk : {} for blk in
                iter_all(Coordinate.FGBlockOnHICANN)},
            'synapse_driver': ValueStorage({
                'stp': None,
                'cap': DAC(0),
                'gmax': DAC(0),
                'gmax_div': DAC(11)
                })
            })

class reset_configuration:
    def __init__(self, fg=False):
        self.fg = fg

    def __call__(self, func):
        fg = self.fg
        def proxy(self, *args, **kwargs):
            self._configured = False
            if fg:
                self._fg_written = False

            return func(self, *args, **kwargs)
        return proxy

def run_configuration(func):
        def proxy(self, *args, **kwargs):
            if not self._configured:

                self.configure(not self._fg_written, not self._fg_written or self._not_reset_count==0)

                if self._not_reset_count >= self._max_without_reset:
                    self._not_reset_count = 0 # reset next time
                else:
                    self._not_reset_count += 1

                self._configured = True
                self._fg_written = True
            return func(self, *args, **kwargs)
        return proxy

def get_bus_from_driver(driver):
        top = driver < 112
        if top:
            return (driver/2)%8
        else:
            return 7 - (driver/2)%8

class FastHICANNConfigurator(HICANNConfigurator):
    def __init__(self, configure_floating_gates=True, reset=True):
        HICANNConfigurator.__init__(self)
        self._configure_floating_gates = configure_floating_gates
        self._reset = reset

    def config_fpga(self, *args, **kwargs):
        if self._reset:
            HICANNConfigurator.config_fpga(self, *args, **kwargs)

    def hicann_init(self, h):
        if self._reset:
            pyhalbe.HICANN.init(h, False)
        #pyhalbe.Support.Power.set_L1_voltages(handle,1.05,1.25)

    def config_floating_gates(self, handle, data):
        if self._configure_floating_gates:
            HICANNConfigurator.config_floating_gates(self, handle, data);

    def config_synapse_array(self, handle, data):
        # if floating gates are configured we also configure synapses
        # should be disentangled
        if self._configure_floating_gates:
            HICANNConfigurator.config_synapse_array(self, handle, data);



class VOLVOHHICANNConfigurator(pysthal.HICANNConfigurator):
    def __init__(self, vol, voh):
        pysthal.HICANNConfigurator.__init__(self)
        self.vol = vol
        self.voh = voh

    def config_fpga(*args):
        pass

    def config(self, fpga, handle, data):
        self.getLogger().info("SETTING VOL: {}, VOH: {}".format(self.vol, self.voh))
        pyhalbe.Support.Power.set_L1_voltages(handle, self.vol, self.voh)


class Hardware(object):
    def __init__(self, wafer, hicann, calibration_backend=None, freq=100e6, bg_period=10000):
        wafer_c = Coordinate.Wafer(wafer)
        hicann_c = Coordinate.HICANNOnWafer(Enum(hicann))

        self.wafer = pysthal.Wafer(wafer_c)
        self.wafer.commonFPGASettings().setPLL(freq)
        self.hicann = self.wafer[hicann_c]

        # Going to be set during routing
        self.sending_link = None
        self.receiving_link = None

        self.params = Parameters()

        self._not_reset_count = 0
        self._max_without_reset = 1500
        self._configured = False
        self._fg_written = False

        # Activate firing for all neurons to work around 1.2V bug
        for nrn in iter_all(Coordinate.NeuronOnHICANN):
            self.hicann.neurons[nrn].activate_firing(True)

        # Configure background generators to fire on L1 address 0
        for bg in iter_all(Coordinate.BackgroundGeneratorOnHICANN):
            generator = self.hicann.layer1[bg]
            generator.enable(True)
            generator.random(False)
            generator.period(bg_period)
            generator.address(BG_ADDRESS)

        if calibration_backend is not None:
            self.calibration = Calibtic(calibration_backend, wafer_c, hicann_c)
        else:
            self.calibration = None

    def set_parameters(self, params):
        if (params.base_parameters != self.params.base_parameters
                or params.neuron_parameters != self.params.neuron_parameters
                or params.shared_parameters != self.params.shared_parameters):
             self._configured = False
             self._fg_written = False
        if (params.synapse_driver != self.params.synapse_driver):
             self._configured = False

        self.params = params

    def setup_driver(self, address, synapse_driver, neuron, line=TOP):
        # Calculate bus from synapse driver
        top = synapse_driver < 112

        bus = get_bus_from_driver(synapse_driver)

        if top:
            if synapse_driver % 2 == 1:
                side = left
            else:
                side = right
        else:
            if synapse_driver % 2 == 0:
                side = left
            else:
                side = right

        # Select neuron
        address = pyhalbe.HICANN.L1Address(address)
        neuron_c = Coordinate.NeuronOnHICANN(Enum(neuron))

        driver_line_c = Coordinate.SynapseSwitchRowOnHICANN(Y(synapse_driver), side)

        # Configure synapse driver
        driver = self.hicann.synapses[Coordinate.SynapseDriverOnHICANN(driver_line_c)]

        driver.set_l1() # Set to process spikes from L1

        if neuron % 2 == 0:
            driver[line].set_decoder(top, address.getDriverDecoderMask())
        else:
            driver[line].set_decoder(bottom, address.getDriverDecoderMask())

        driver[line].set_syn_in(left, 1)
        driver[line].set_syn_in(right, 0)

        # Configure synaptic inputs
        synapse_line_c = Coordinate.SynapseRowOnHICANN(Coordinate.SynapseDriverOnHICANN(driver_line_c.line(), side), line)

        synapse_line = self.hicann.synapses[synapse_line_c]

        # … for first neuron
        synapse_line.weights[int(neuron_c.x())] = HICANN.SynapseWeight(15) # Set weights
        synapse_line.decoders[int(neuron_c.x())] = HICANN.SynapseDecoder(address.getSynapseDecoderMask())

    @reset_configuration()
    def add_route(self, address, synapse_driver, neuron, line=TOP):
        # Calculate bus from synapse driver
        top = synapse_driver < 112

        bus = get_bus_from_driver(synapse_driver)

        if top:
            if synapse_driver % 2 == 1:
                side = left
            else:
                side = right
        else:
            if synapse_driver % 2 == 0:
                side = left
            else:
                side = right

        # Select neuron
        address = pyhalbe.HICANN.L1Address(address)
        neuron_c = Coordinate.NeuronOnHICANN(Enum(neuron))

        # Setup Gbit Links and merger tree
        self.sending_link = Coordinate.GbitLinkOnHICANN(bus)
        self.hicann.layer1[self.sending_link] = pyhalbe.HICANN.GbitLink.Direction.TO_HICANN

        for merger in iter_all(Coordinate.Merger0OnHICANN):
            merger = self.hicann.layer1[merger]
#            merger.slow = True

        for merger in iter_all(Coordinate.Merger1OnHICANN):
            merger = self.hicann.layer1[merger]
#            merger.slow = True

        for merger in iter_all(Coordinate.Merger2OnHICANN):
            merger = self.hicann.layer1[merger]
#            merger.slow = True

        for merger in iter_all(Coordinate.Merger3OnHICANN):
            merger = self.hicann.layer1[merger]
#            merger.slow = True

        for merger in iter_all(Coordinate.DNCMergerOnHICANN):
            merger = self.hicann.layer1[merger]
            merger.slow = True
            merger.loopback = False

        # Configure sending repeater to forward spikes to the right
        h_line = Coordinate.HLineOnHICANN(2 * 4 * (8 - bus) - 2)
        sending_repeater = self.hicann.repeater[Coordinate.HRepeaterOnHICANN(h_line, left)]
        sending_repeater.setOutput(right, True)

        # Enable a crossbar switch to route the signal into the first vertical line
        if side == left:
            v_line_c = Coordinate.VLineOnHICANN(4*bus + 32)
        else:
            v_line_c = Coordinate.VLineOnHICANN(159 - 4*bus + 32)

        self.hicann.crossbar_switches.set(v_line_c, h_line, True)
         
        # Configure synapse switches and forward to synapse switch row 81
        driver_line_c = Coordinate.SynapseSwitchRowOnHICANN(Y(synapse_driver), side)
        self.hicann.synapse_switches.set(v_line_c, driver_line_c.line(), True)
         
        # Configure synapse driver
        driver = self.hicann.synapses[Coordinate.SynapseDriverOnHICANN(driver_line_c)]
         
        driver.set_l1() # Set to process spikes from L1
        
        if neuron % 2 == 0:
            driver[line].set_decoder(top, address.getDriverDecoderMask())
        else:
            driver[line].set_decoder(bottom, address.getDriverDecoderMask())

        driver[line].set_syn_in(left, 1)
        driver[line].set_syn_in(right, 0)

        # Configure synaptic inputs
        synapse_line_c = Coordinate.SynapseRowOnHICANN(Coordinate.SynapseDriverOnHICANN(driver_line_c.line(), side), line)

        synapse_line = self.hicann.synapses[synapse_line_c]

        # … for first neuron
        synapse_line.weights[int(neuron_c.x())] = HICANN.SynapseWeight(15) # Set weights
        synapse_line.decoders[int(neuron_c.x())] = HICANN.SynapseDecoder(address.getSynapseDecoderMask())

        return bus, h_line, v_line_c

    def enable_readout(self, dnc):
        dnc_link = Coordinate.GbitLinkOnHICANN(dnc)
        self.hicann.layer1[dnc_link] = pyhalbe.HICANN.GbitLink.Direction.TO_DNC

    def assign_address(self, neuron, address):
        self.hicann.enable_l1_output(Coordinate.NeuronOnHICANN(Enum(neuron)), HICANN.L1Address(address))
        self.hicann.neurons[Coordinate.NeuronOnHICANN(Enum(neuron))].activate_firing(True)

    @reset_configuration()
    def clear_routes(self):
        self.hicann.clear_l1_routing()
        self.hicann.clear_l1_switches()
        self.hicann.synapses.clear_drivers()

        self.hicann.clear_complete_l1_routing()

        #self.hicann.synapses.clear_synapses() # make me configurable

        for nrn in iter_all(Coordinate.NeuronOnHICANN):
            self.hicann.neurons[nrn].enable_spl1_output(False)
            self.hicann.neurons[nrn].activate_firing(False)

        # enable all horizontal repeaters
        #for r_c in Coordinate.iter_all(Coordinate.HRepeaterOnHICANN):
        #    self.hicann.repeater[r_c].setOutput(Coordinate.right, True)

    @reset_configuration()
    def enable_adc(self, neuron, adc):
        neuron_c = Coordinate.NeuronOnHICANN(Enum(neuron))
        self.hicann.enable_aout(neuron_c, Coordinate.AnalogOnHICANN(adc))

    def connect(self):
        # Connect to hardware
        connection_db = pysthal.MagicHardwareDatabase()
        self.wafer.connect(connection_db)

    def disconnect(self):
        self.wafer.disconnect()

    def clear_spike_trains(self):
        self.wafer.clearSpikes()

    def add_spike_train(self, bus, address, times):
        # Construct spike train
        spikes = pysthal.Vector_Spike()
        for t in times:
            spikes.append(pysthal.Spike(pyhalbe.HICANN.L1Address(address), t))
        sending_link = Coordinate.GbitLinkOnHICANN(bus)
        self.hicann.sendSpikes(sending_link, spikes)

    def configure(self, write_floating_gates=True, reset=True):
        s = time.time()
        # Configure synapse drivers
        for driver_c in iter_all(Coordinate.SynapseDriverOnHICANN):
            driver = self.hicann.synapses[driver_c]
            for i in [TOP, BOT]:
                driver[i].set_gmax(self.params.synapse_driver.gmax.toDAC().value)
                driver[i].set_gmax_div(left, self.params.synapse_driver.gmax_div.toDAC().value)
                driver[i].set_gmax_div(right, self.params.synapse_driver.gmax_div.toDAC().value)

            if self.params.synapse_driver.stp == 'facilitation':
                driver.set_stf()
            elif self.params.synapse_driver.stp == 'depression':
                driver.set_std()

            driver.stp_cap = self.params.synapse_driver.cap.toDAC().value

        # Now we can set the floating gate parameters for the neurons
        if write_floating_gates:
            for neuron in iter_all(Coordinate.NeuronOnHICANN):
                params = copy.deepcopy(self.params.base_parameters)
                params.update(self.params.neuron_parameters[neuron])
                self.calibration.set_neuron_parameters(
                    params, neuron, self.hicann.floating_gates)
            for block in  iter_all(Coordinate.FGBlockOnHICANN):
                params = copy.deepcopy(self.params.base_parameters)
                params.update(self.params.shared_parameters[block])
                self.calibration.set_shared_parameters(
                    params, block, self.hicann.floating_gates)
            print(self.hicann.floating_gates)


        # Write configuration
        self.wafer.configure(FastHICANNConfigurator(write_floating_gates, reset))
        #self.wafer.configure(pysthal.HICANNConfigurator())
    
    @run_configuration
    def record(self, adc, duration):
        # Setup ADC
        recorder0 = self.hicann.analogRecorder(Coordinate.AnalogOnHICANN(adc))
        recorder0.activateTrigger(duration)

        # Run experiment:
        runner = pysthal.ExperimentRunner(duration)
        self.wafer.start(runner)

        # Return voltage trace
        trace = recorder0.trace()
        timestamps = recorder0.getTimestamps()
        recorder0.freeHandle()
        return timestamps, trace

    @run_configuration
    def run(self, duration):
        time.sleep(1.0)
        runner = pysthal.ExperimentRunner(duration)
        self.wafer.start(runner)

    def get_spikes(self, bus):
        bus = Coordinate.GbitLinkOnHICANN(bus)
        spikes = self.hicann.receivedSpikes(bus)
        return spikes


