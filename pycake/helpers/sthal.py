"""Helper classes for StHAL."""

import math
import os
import hashlib
import cPickle
import pyhalbe
import pysthal
import pylogging
import Coordinate
from pyhalbe import HICANN
from Coordinate import Enum, X, Y
from Coordinate import NeuronOnHICANN
from Coordinate import BackgroundGeneratorOnHICANN
from Coordinate import SynapseDriverOnHICANN
from Coordinate import GbitLinkOnHICANN
from sims.sim_denmem_lib import NETS_AND_PINS
from sims.sim_denmem_lib import TBParameters
from sims.sim_denmem_lib import run_remote_simulation


class SpikeReadoutHICANNConfigurator(pysthal.HICANNConfigurator):

    def config_fpga(self, fpga_handle, fpga):

        #self.config_dnc_link(fpga_handle, fpga)

        pass

    def config(self, fpga, handle, data):

        pyhalbe.HICANN.init(handle, False)

        #self.config_floating_gates(handle, data)
        self.config_fg_stimulus(handle, data)

        #self.config_synapse_array(handle, data)

        self.config_neuron_quads(handle, data)
        self.config_phase(handle, data)
        self.config_gbitlink(handle, data)

        #self.config_synapse_drivers(handle, data)
        self.config_synapse_switch(handle, data)
        self.config_stdp(handle, data)
        self.config_crossbar_switches(handle, data)
        self.config_repeater(handle, data)
        self.config_merger_tree(handle, data)
        self.config_dncmerger(handle, data)
        self.config_background_generators(handle, data)
        self.flush_fpga(fpga)
        self.lock_repeater(handle, data)

        self.config_neuron_config(handle, data)
        self.config_neuron_quads(handle, data)
        self.config_analog_readout(handle, data)
        self.flush_fpga(fpga)


class UpdateAnalogOutputConfigurator(pysthal.HICANNConfigurator):
    """ Configures the following things from sthal container:
        - neuron quad configuration
        - analog readout
        - analog current input
        - current stimulus strength/duration etc.
    """
    def config_fpga(self, *args):
        """do not reset FPGA"""
        pass

    def config(self, fpga_handle, h, hicann):
        """Call analog output related configuration functions."""
        self.config_neuron_config(h, hicann)
        self.config_neuron_quads(h, hicann)
        self.config_analog_readout(h, hicann)
        self.config_fg_stimulus(h, hicann)
        self.flush_fpga(fpga_handle)


class UpdateParameter(pysthal.HICANNConfigurator):
    def __init__(self, neuron_parameters):
        pysthal.HICANNConfigurator.__init__(self)
        self.blocks = dict(
            (row, []) for row in Coordinate.iter_all(Coordinate.FGRowOnFGBlock))
        for parameter in neuron_parameters:
            for block in Coordinate.iter_all(Coordinate.FGBlockOnHICANN):
                self.blocks[HICANN.getNeuronRow(block, parameter)].append(block)

    def hicann_init(self, h):
        HICANN.init(h, False)

    def config_synapse_array(self, handle, data):
        pass

    def write_fg_row(self, handle, row, fg, write_down):
        blocks = self.blocks[row]
        if len(blocks) == 4:
            pysthal.HICANNConfigurator.write_fg_row(
                self, handle, row, fg, write_down)
        else:
            for block in blocks:
                data = fg.getBlock(block).getFGRow(row)
                self.getLogger().info("Update {} on {}".format(row, block))
                HICANN.set_fg_row_values(handle, block, row, data, write_down)


class FakeAnalogRecorder(object):
    """Fake AnalogRecorder for testing purposes.

    Mimics API from StHAL::AnalogRecorder.
    """
    def __init__(self):
        pass

    def freeHandle(self):
        pass

    def record(self):
        pass

    def trace(self):
        pass

    def setRecordingTime(self, time):
        pass

    def getTimestamps(self):
        pass

    def status(self):
        pass


class StHALContainer(object):
    """Contains StHAL objects for hardware access. Multiple experiments can share one container."""
    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['hicann']
        return odict

    def __setstate__(self, dic):
        coord_hicann = dic['coord_hicann']
        wafer = dic['wafer']
        dic['hicann'] = wafer[coord_hicann]
        self.__dict__.update(dic)

    logger = pylogging.get("pycake.helper.sthal")

    def __init__(self, coord_wafer,
                 coord_hicann,
                 coord_analog=Coordinate.AnalogOnHICANN(0),
                 recording_time=1.e-4,
                 wafer_cfg="",
                 PLL=100e6,
                 dump_file=None,
                 neuron_size=1):
        """Initialize StHAL. kwargs default to vertical setup configuration.

        Args:
            coord_hicann: HICANN Coordinate
            coord_analog: AnalogOnHICANN Coordinate
            recording_time: ADC recording time in seconds
            wafer_cfg: ?
            PLL: HICANN PLL frequency in Hz
            dump_file: filename for StHAL dump handle instead of hardware
        """

        self.coord_wafer = coord_wafer
        self.coord_hicann = coord_hicann
        self.dump_file = dump_file

        self.wafer_cfg = wafer_cfg

        self.wafer = pysthal.Wafer(coord_wafer)  # Stateful HICANN Container

        if self.wafer_cfg:
            self.logger.info("Loading {}".format(wafer_cfg))
            self.wafer.load(wafer_cfg)

        self.setPLL(PLL)
        self.hicann = self.wafer[coord_hicann]
        self.adc = None
        self.recording_time = recording_time
        self.coord_analog = coord_analog
        self._connected = False
        self.input_spikes = {}

    def connect(self):
        """Connect to the hardware."""
        if self.dump_file is None:
            db = pysthal.MagicHardwareDatabase()
            self.wafer.connect(db)
        else:
            # do not actually connect, write to file
            self.dump_connect(self.dump_file, False)
        self.connect_adc()
        self._connected = True

    def dump_connect(self, filename, append):
        self.wafer.connect(pysthal.DumpHardwareDatabase(filename, append))
        self._connected = True

    def connect_adc(self, coord_analog=None):
        """Gets ADC handle.

        Args:
            coord_analog: Coordinate.AnalogOnHICANN to override default behavior
        """
        # analogRecoorder() MUST be called after wafer.connect()
        if self.adc:
            self.adc.freeHandle()
            self.adc = None

        if coord_analog is None:
            coord_analog = self.coord_analog

        if self.dump_file is None:
            adc = self.hicann.analogRecorder(coord_analog)
        else:
            # do not actually connect
            adc = FakeAnalogRecorder()
        adc.setRecordingTime(self.recording_time)
        self.adc = adc

    def disconnect(self):
        """Free handles."""
        self.wafer.disconnect()
        self.adc.freeHandle()
        self.adc = None
        self._connected = False

    # TODO fix after dominiks thesis
    def write_config(self, program_floating_gates=True, configurator=None):
        """Write full configuration."""
        if not self._connected:
            self.connect()
        if configurator is None:
            if program_floating_gates:
                configurator = pysthal.HICANNConfigurator()
            else:
                configurator = pysthal.DontProgramFloatingGatesHICANNConfigurator()

        for _ in range(3):
            try:
                # self.set_fg_biasn(0)
                self.wafer.configure(configurator)
                return
            except RuntimeError as err:
                if err.message == 'fg_log_error timeout':
                    self.logger.warn("Error configuring floating gates, retry")
                    continue
                else:
                    raise
        msg = "Maximum retry count for configuration exceded"
        self.logger.error(msg)
        raise RuntimeError(msg)

    def switch_analog_output(self, coord_neuron, l1address=None):
        """Write analog output configuration (only).
           If l1address is None, l1 output is disabled.

           Using write_config() after this function will result
           in errors like
           HostALController: FPGA expected sequence number 00000000 instead of the current 000xxxxx. Cannot correct this.
           HostALController::sendSingleFrame: did not get an answer from FPGA.
        """
        self.hicann.disable_aout()
        self.hicann.disable_l1_output()
        self.hicann.disable_current_stimulus()
        if not self.wafer_cfg and l1address is not None:
            self.hicann.enable_l1_output(coord_neuron, pyhalbe.HICANN.L1Address(l1address))
        self.hicann.enable_aout(coord_neuron, self.coord_analog)
        self.write_config(configurator=UpdateAnalogOutputConfigurator())

    def send_spikes_to_all_neurons(self, spike_times, excitatory=True,
                                   gmax_div=2, locking_freq=0.05e6):
        """Stimulates all neurons with the given spike train"""
        assert(locking_freq <= 5.0e6)

        self.hicann.clear_complete_l1_routing()

        stim_l1address = pyhalbe.HICANN.L1Address(63)
        lock_l1address = pyhalbe.HICANN.L1Address(0)
        top = (GbitLinkOnHICANN(1), SynapseDriverOnHICANN(Enum(99)))
        bottom = (GbitLinkOnHICANN(0), SynapseDriverOnHICANN(Enum(126)))

        spikes = pysthal.Vector_Spike([
            pysthal.Spike(stim_l1address, t) for t in spike_times])

        PLL = self.getPLL()
        bg_period = int(math.floor(PLL/locking_freq) - 1)
        for link, driver in (top, bottom):
            bg = BackgroundGeneratorOnHICANN(int(link))

            generator = self.hicann.layer1[bg]
            generator.enable(True)
            generator.random(False)
            generator.period(bg_period)
            generator.address(lock_l1address)
            self.logger.DEBUG("activate {!s} with period {}".format(bg, bg_period))

            self.hicann.route(link.toOutputBufferOnHICANN(), driver)
            self.enable_synapse_line(
                driver, stim_l1address, gmax_div, excitatory)

            self.hicann.layer1[link] = HICANN.GbitLink.TO_HICANN

            self.hicann.sendSpikes(link, spikes)

    def read_adc(self):
        """Run experiment, read out ADC.
        """
        if not self._connected:
            self.connect()
        max_tries = 10
        for ii in range(max_tries):
            try:
                self.adc.record()
                return {'t': self.adc.getTimestamps(), 'v': self.adc.trace()}
            except RuntimeError as e:
                print e
                print "retry"
                self.connect_adc()
        raise RuntimeError("Aborting ADC readout, maximum number of retries exceded")

    def read_adc_and_spikes(self):
        """Run pysthal.ExperimentRunner, read back ADC trace and spikes"""
        max_tries = 10
        for ii in range(max_tries):
            try:
                self._sendSpikes(self.input_spikes)
                runtime = self.adc.getRecordingTime() + 1.e-5
                runner = pysthal.ExperimentRunner(runtime)
                self.wafer.restart(runner) # Clears received spikes
                self.adc.record()  # TODO triggered
                recv_spikes = {}
                for link in Coordinate.iter_all(GbitLinkOnHICANN):
                    recv_spikes[link] = self.hicann.receivedSpikes(link)
                return {'t': self.adc.getTimestamps(),
                        'v': self.adc.trace(),
                        's': recv_spikes}
            except RuntimeError as e:
                print e
                print "retry"
                self.connect_adc()
        raise RuntimeError("Aborting ADC readout, maximum number of retries exceded")


    def read_adc_status(self):
        if not self._connected:
            self.connect()
        return self.adc.status()

    def read_wafer_status(self):
        if not self._connected:
            self.connect()
        return self.wafer.status()

    def getPLL(self):
        return self.wafer.commonFPGASettings().getPLL()

    def setPLL(self, freq):
        """
        pll frequency of the hicann, musst be 50.0e6, 100.0e6, 150.0e6, 200.0e6 or 250.0e6
        """
        self.logger.info("Setting PLL to {} MHz".format(freq/1e6))
        return self.wafer.commonFPGASettings().setPLL(freq)

    def set_fg_biasn(self, biasn):
        fg = self.hicann.floating_gates
        for ii in range(fg.getNoProgrammingPasses()):
            cfg = fg.getFGConfig(Enum(ii))
            cfg.fg_biasn = 0
            fg.setFGConfig(Enum(ii), cfg)

    def stimulateNeurons(self, rate, no_generators, excitatory=True):
        """Stimulate neurons via background generators

        Args:
            rate: Rate of a single generator in Hertz
            number: Number of generators to use in parallel (per Neuron)
        """
        assert(no_generators >= 0 and no_generators <= 4)
        assert(rate <= 5.0e6)

        l1address = pyhalbe.HICANN.L1Address(0)

        PLL = self.getPLL()
        bg_period = int(math.floor(PLL/rate) - 1)
        self.logger.info("Stimulating neurons from {} background generators"
                         " with isi {}".format(no_generators, bg_period))

        for bg in Coordinate.iter_all(Coordinate.BackgroundGeneratorOnHICANN):
            generator = self.hicann.layer1[bg]
            generator.enable(bg.value()/2 < no_generators)
            generator.random(False)
            generator.period(bg_period)
            generator.address(l1address)
            self.logger.DEBUG("activate {!s} with period {}".format(bg, bg_period))

        links = []
        for ii in range(4):
            bg_top = Coordinate.OutputBufferOnHICANN(2*ii+1)
            bg_bottom = Coordinate.OutputBufferOnHICANN(2*ii)
            links.append(GbitLinkOnHICANN(2 * ii + 1))
            links.append(GbitLinkOnHICANN(2 * ii))

            drv_top = Coordinate.SynapseDriverOnHICANN(
                Coordinate.Enum(99 + ii * 4))
            drv_bottom = Coordinate.SynapseDriverOnHICANN(
                Coordinate.Enum(126 - ii * 4))
            if ii < no_generators:
                self.route(bg_top, drv_top)
                self.route(bg_bottom, drv_bottom)
                self.enable_synapse_line(drv_top, l1address, 2, excitatory)
                self.enable_synapse_line(drv_bottom, l1address, 2, excitatory)
            else:
                self.disable_synapse_line(drv_top)
                self.disable_synapse_line(drv_bottom)
        return links

    def configure_synapse_driver(self, driver_c, l1address, gmax_div, gmax=0):
        """
        Configures top row of the synapse driver to give exitatory input and
        bottom row to give inhibitory input

        Arguments:
            driver_c: [SynapseDriverOnHICANN] driver to configure
            address: [HICANN.L1Address] l1 address of spikes
            gmax_div: Gmax divider in range 2..30
            gmax: gmax fg value to use, default 0
        """
        assert(2 <= gmax_div <= 30)

        from Coordinate import left, right, top, bottom

        driver = self.hicann.synapses[driver_c]
        driver_decoder = l1address.getDriverDecoderMask()
        driver.set_l1()
        driver[top].set_decoder(top, driver_decoder)
        driver[top].set_decoder(bottom, driver_decoder)
        driver[top].set_gmax(gmax)
        # There are to parallel circuits for the divisor in the synapse
        # driver (left, right), the sum of both divisor values gives the
        # actual divisor value applied to vgmax
        driver[top].set_gmax_div(left, gmax_div / 2)
        driver[top].set_gmax_div(right, gmax_div / 2 + gmax_div % 2)

        # copy config above
        driver[bottom] = driver[top]

        # Configure top to exictatory inputs and bottom to inhibitory
        driver[top].set_syn_in(left, 1)  # Esynx
        driver[top].set_syn_in(right, 0)  # Esyni, perhaps
        driver[bottom].set_syn_in(left, 0)
        driver[bottom].set_syn_in(right, 1)

    def enable_synapse_line(self, driver_c, l1address, gmax_div=2, excitatory=True):
        """
        """
        from Coordinate import top, bottom

        self.configure_synapse_driver(driver_c, l1address, gmax_div, 0)

        if excitatory:
            w_top = [pyhalbe.HICANN.SynapseWeight(15)] * 256
            w_bottom = [pyhalbe.HICANN.SynapseWeight(0)] * 256
        else:
            w_top = [pyhalbe.HICANN.SynapseWeight(0)] * 256
            w_bottom = [pyhalbe.HICANN.SynapseWeight(15)] * 256

        synapse_line_top = Coordinate.SynapseRowOnHICANN(driver_c, top)
        synapse_line_bottom = Coordinate.SynapseRowOnHICANN(driver_c, bottom)
        self.hicann.synapses[synapse_line_top].weights[:] = w_top
        self.hicann.synapses[synapse_line_bottom].weights[:] = w_bottom
        synapse_decoder = [l1address.getSynapseDecoderMask()] * 256
        self.hicann.synapses[synapse_line_top].decoders[:] = synapse_decoder
        self.hicann.synapses[synapse_line_bottom].decoders[:] = synapse_decoder
        self.logger.DEBUG("enabled {!s} listing to {!s}".format(
            driver_c, l1address))

    def disable_synapse_line(self, driver_c):
        """
        """

        top = Coordinate.top
        bottom = Coordinate.bottom

        driver = self.hicann.synapses[driver_c]
        driver.disable()
        synapse_line_top = Coordinate.SynapseRowOnHICANN(driver_c, top)
        synapse_line_bottom = Coordinate.SynapseRowOnHICANN(driver_c, bottom)
        weights = [pyhalbe.HICANN.SynapseWeight(0)] * 256
        self.hicann.synapses[synapse_line_top].weights[:] = weights
        self.hicann.synapses[synapse_line_bottom].weights[:] = weights
        self.logger.DEBUG("disabled {!s}".format(driver_c))

    def route(self, output_buffer, driver, route=0):
        """Connects one output buffer to a synapse driver via l1 routing

        Disclaimer: not testet for synapse drivers on the right side!
        """

        assert(route >= 0 and route <= 4)
        repeater = output_buffer.toSendingRepeaterOnHICANN().toHRepeaterOnHICANN()
        out_line = output_buffer.toSendingRepeaterOnHICANN().toHRepeaterOnHICANN().toHLineOnHICANN()
        driver_line = driver.line()

        repeater_data = self.hicann.repeater[repeater]
        repeater_data.setOutput(Coordinate.right, True)
        if driver.toSideHorizontal() == Coordinate.left:
            v_line_value = 31 - out_line.value()/2
            v_line_value += 32 * route
        if driver.toSideHorizontal() == Coordinate.right:
            v_line_value = 128 + out_line.value()/2
            v_line_value += 32 * route
        v_line = Coordinate.VLineOnHICANN(v_line_value)
        chain = (output_buffer, repeater, out_line, v_line, driver_line, driver)
        dbg = " -> ".join(['{!s}'] * len(chain))
        self.logger.DEBUG("connected " + dbg.format(*chain))
        self.hicann.crossbar_switches.set(v_line, out_line, True)
        self.hicann.synapse_switches.set(v_line, driver_line, True)

    def set_current_stimulus(self, stim_value, stim_length, pulse_length=15):
        """ Set current. Does not write to hardware.

            Args:
                stim_value: int -> strength of current (DAC -> nA conversion
                            unclear, but ~ 2nA per 500 DAC)
                stim_length: int -> how long should the current be?
                             max. duration is a whole cycle: 129

            Returns:
                length of one pulse in seconds
        """
        stimulus = pysthal.FGStimulus()
        stimulus.setPulselength(pulse_length)
        stimulus.setContinuous(True)

        stimulus[:stim_length] = [stim_value] * stim_length
        stimulus[stim_length:] = [0] * (len(stimulus) - stim_length)

        for block in range(4):
            self.hicann.current_stimuli[block] = stimulus

        return (pulse_length+1)*4 * stim_length / self.getPLL()

    def switch_current_stimulus_and_output(self, coord_neuron, l1address=None):
        """ Switches the current stimulus and analog output to a certain neuron.
            To avoid invalid neuron configurations (see HICANN doc page 33),
            all aouts and current stimuli are disabled before enabling them for
            one neuron."""
        if not self._connected:
            self.connect()
        self.hicann.disable_aout()
        self.hicann.disable_current_stimulus()
        self.hicann.enable_current_stimulus(coord_neuron)
        if not l1address is None:
            self.hicann.enable_l1_output(coord_neuron, pyhalbe.HICANN.L1Address(l1address))
        self.hicann.enable_aout(coord_neuron, self.coord_analog)
        self.wafer.configure(UpdateAnalogOutputConfigurator())

    def set_recording_time(self, recording_time, repeations):
        """Sets the recording time of the ADC.

        The recording_time should be as small as theoretical required for the
        measurement. And the repeations factor should be the amount you need
        to cope with readout noise. This is to speed up simulations
        """

        self.recording_time = recording_time * repeations

    def set_neuron_size(self, n):
        self.logger.INFO("Setting neuron size to {}".format(n))
        self.firing_denmems = list(self.hicann.set_neuron_size(n))

class SimStHALContainer(StHALContainer):
    """Contains StHAL objects for hardware access. Multiple experiments can 
    share one container.

    Attributes:
        remote_host: hostname of simulation server
        remote_port: port of simulation server
        hicann_version: hicann version to simulate (2,4)
        mc_seed: Monte carlo seed (None disables MC)
        maximum_spikes: abort simulation after this many spikes (default=200)
        spike_counter_offset: start counting spikes after this time (default
                              simulation_init_time)
    """
    logger = pylogging.get("pycake.helper.simsthal")

    def __init__(self, coord_wafer,
                 coord_hicann,
                 coord_analog=Coordinate.AnalogOnHICANN(0),
                 recording_time=30.0e-6,
                 wafer_cfg="",
                 PLL=100e6,
                 dump_file=None,
                 config=None,
                 neuron_size=1):
        """Initialize StHAL. kwargs default to vertical setup configuration.

        Args:
            coord_hicann: HICANN Coordinate
            coord_analog: AnalogOnHICANN Coordinate
            recording_time: ADC recording time in seconds
            wafer_cfg: ?
            PLL: HICANN PLL frequency in Hz
            dump_file: filename for StHAL dump handle instead of hardware
            remote_host: host
            remote_port: port
        """
        super(SimStHALContainer, self).__init__(
            coord_wafer, coord_hicann, coord_analog, recording_time,
            wafer_cfg, PLL, dump_file, neuron_size)
        self.current_neuron = Coordinate.NeuronOnHICANN()
        self.recorded_traces = {}
        host, port = config.get_sim_denmem().split(':')
        self.remote_host = host
        self.remote_port = int(port)

        # 10 times simulation reset
        self.simulation_init_time = 10.0e-07
        self.set_recording_time(recording_time, 1)

        self.logger.INFO("Using sim_denmem on {}:{}".format(
            self.remote_host, self.remote_port))

        self.simulation_cache = config.get_sim_denmem_cache()
        if self.simulation_cache and not os.path.isdir(self.simulation_cache):
            raise RuntimeError("simulation_cache must be a folder")
        self.hicann_version = config.get_hicann_version()
        self.mc_seed = config.get_sim_denmem_mc_seed()

        self.maximum_spikes = 200
        self.spike_counter_offset = self.simulation_init_time

    def connect(self):
        """Connect to the hardware."""
        pass

    def dump_connect(self, filename, append):
        raise NotImplementedError()

    def connect_adc(self, coord_analog=None):
        """Gets ADC handle.

        Args:
            coord_analog: Coordinate.AnalogOnHICANN to override default behavior
        """
        pass

    def disconnect(self):
        """Free handles."""
        for k in self.recorded_traces.keys():
            self.recorded_traces[k] = None

    def write_config(self, program_floating_gates=True, configurator=None):
        """Write full configuration."""
        pass

    def switch_analog_output(self, coord_neuron, l1address=None):
        super(SimStHALContainer, self).switch_analog_output(
            coord_neuron, l1address)
        self.current_neuron = coord_neuron

    def read_adc(self):
        """Fake ADC readout by evaluating a denmem_sim run."""

        param = self.build_TBParameters(self.current_neuron)
        json = param.to_json()
        key = (self.current_neuron, json)
        # After finishing the simulation the recorded traces are set to None
        # in order to preserve memory and diskspace.
        # They will be stored by the Measurement class, if save_traces is set
        if self.recorded_traces.get(key, None) is None:
            self.run_simulation(self.current_neuron, param)
        return self.recorded_traces[key]

    def read_adc_and_spikes(self):
        return self.read_adc()

    def get_simulation_neurons(self, neuron):
        """Returns the two neurons, that will be simulated for a given neuron"""
        if int(neuron.x()) % 2 == 0:
            return neuron, NeuronOnHICANN(X(int(neuron.x()) + 1), neuron.y())
        else:
            return NeuronOnHICANN(X(int(neuron.x()) - 1), neuron.y()), neuron

    def build_TBParameters(self, neuron):
        """Returns the serialized TBParameters for self.current_neuron"""
        left, right = self.get_simulation_neurons(neuron)
        param = TBParameters.from_sthal(self.wafer, self.coord_hicann, left,
                                        self.simulation_init_time,
                                        self.spike_counter_offset)
        param.simulator_settings.simulation_time = self.recording_time
        param.simulator_settings.nets_to_save = NETS_AND_PINS.ALL
        param.simulator_settings.hicann_version = self.hicann_version
        param.simulator_settings.max_spike_count = self.maximum_spikes

        # HACK, enable analog output for both neurons, if no current input
        # is enabled. Otherwise caching would not work as expected.
        if param.digital_parameters["iout"][2] is False:
            param.digital_parameters["iout"] = (True, True, False)

        if self.mc_seed is not None:
            mc_run = int(neuron.id())/2 + 1
            param.simulator_settings.set_mc_run(self.mc_seed, mc_run)

        return param

    def run_simulation(self, neuron, param):
        """Execute a remote simulation for the given json set"""
        # TODO Error handling
        json = param.to_json()
        json_hash = None
        left, right = self.get_simulation_neurons(neuron)

        if self.simulation_cache:
            json_hash = hashlib.new('sha256')
            json_hash.update(json)
            json_hash = os.path.join(
                self.simulation_cache, json_hash.hexdigest())

        if json_hash and os.path.isfile(json_hash):
            self.logger.info("load result from cache: {}".format(json_hash))
            with open(json_hash) as infile:
                json_loaded, lresult, rresult = cPickle.load(infile)
                assert json_loaded == json
        else:
            lresult, rresult = run_remote_simulation(
                param, self.remote_host, self.remote_port,
                init_time=self.simulation_init_time)

            if json_hash:
                self.logger.info("cache result in {}".format(json_hash))
                with open(json_hash, 'w') as outfile:
                    data = (json, lresult, rresult)
                    cPickle.dump(data, outfile, cPickle.HIGHEST_PROTOCOL)

        self.recorded_traces[(left, json)] = lresult
        self.recorded_traces[(right, json)] = rresult

    def read_adc_status(self):
        return "FAKE ADC :P"

    def read_wafer_status(self):
        return self.wafer.status()

    def set_recording_time(self, recording_time, _):
        """Sets the recording time of the ADC.

        The recording_time should be as small as theoretical required for the
        measurement. And the repeations factor should be the amount you need
        to cope with readout noise. This is to speed up simulations.

        To speed up simulations this implementation ignores repeations factor!
        """
        self.recording_time = recording_time + self.simulation_init_time
        self.spike_counter_offset = self.simulation_init_time

    def set_neuron_size(self, n):
        self.logger.ERROR("Neuron size other than 1 not supported! Using size 1")
