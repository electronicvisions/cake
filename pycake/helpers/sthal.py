"""Helper classes for StHAL."""

import math
import pyhalbe
import pysthal
import pylogging
from pyhalbe import Coordinate

class UpdateAnalogOutputConfigurator(pysthal.HICANNConfigurator):
    """Change analog output only without writing other configuration."""
    def config_fpga(self, *args):
        """do not reset FPGA"""
        pass

    def config(self, fpga_handle, h, hicann):
        """Call analog output related configuration functions."""
        self.config_neuron_quads(h, hicann)
        self.config_analog_readout(h, hicann)
        self.flush_fpga(fpga_handle)

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
                 recording_time=1.e-4):
        """Initialize StHAL. kwargs default to vertical setup configuration."""

        self.coord_wafer = coord_wafer
        self.coord_hicann = coord_hicann

        wafer = pysthal.Wafer(coord_wafer)  # Stateful HICANN Container
        hicann = wafer[coord_hicann]

        self.wafer = wafer
        self.hicann = hicann
        self.adc = None
        self.recording_time = recording_time
        self.coord_analog = coord_analog
        self._connected = False

    def connect(self):
        """Connect to the hardware."""
        self.wafer.connect(pysthal.MagicHardwareDatabase())
        self.connect_adc()
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
        adc = self.hicann.analogRecorder(coord_analog)
        adc.setRecordingTime(self.recording_time)
        self.adc = adc

    def disconnect(self):
        """Free handles."""
        self.wafer.disconnect()
        self.adc.freeHandle()
        self.adc = None
        self._connected = False

    def write_config(self):
        """Write full configuration."""
        if not self._connected:
            self.connect()
        self.wafer.configure(pysthal.HICANNConfigurator())

    def switch_analog_output(self, coord_neuron, l1address=0, analog=0):
        """Write analog output configuration (only)."""
        if not self._connected:
            self.connect()
        self.hicann.enable_l1_output(coord_neuron, pyhalbe.HICANN.L1Address(l1address))
        self.hicann.enable_aout(coord_neuron, Coordinate.AnalogOnHICANN(analog))
        self.wafer.configure(UpdateAnalogOutputConfigurator())

    def read_adc(self):
        if not self._connected:
            self.connect()
        max_tries = 10
        for ii in range(max_tries):
            try:
                self.adc.record()
                return self.adc.getTimestamps(), self.adc.trace()
            except RuntimeError as e:
                print e
                print "retry"
                self.connect_adc()
        raise RuntimeError("Aborting ADC readout, maximum number of retries exceded")

    def status(self):
        if not self._connected:
            self.connect()
        return self.wafer.status()

    def stimulateNeurons(self, rate, no_generators):
        """Stimulate neurons via background generators

        Args:
            rate: Rate of a single generator in Hertz
            number: Number of generators to use in parallel (per Neuron)
        """
        assert(no_generators>= 0 and no_generators <= 4)
        assert(rate <= 5.0e6)

        l1address = pyhalbe.HICANN.L1Address(0)
        bg_period = int(math.floor(self.hicann.pll_freq/rate) - 1)

        for bg in Coordinate.iter_all(Coordinate.BackgroundGeneratorOnHICANN):
            generator = self.hicann.layer1[bg]
            generator.enable(bg.value()/2 < no_generators)
            generator.random(False)
            generator.period(bg_period)
            generator.address(l1address)
            self.logger.DEBUG("activate {!s} with period {}".format(bg, bg_period))

        for ii in range(4):
            bg_top    = Coordinate.OutputBufferOnHICANN(2*ii+1)
            bg_bottom = Coordinate.OutputBufferOnHICANN(2*ii)

            drv_top    = Coordinate.SynapseDriverOnHICANN(
                    Coordinate.Enum( 99 + ii * 4))
            drv_bottom = Coordinate.SynapseDriverOnHICANN(
                    Coordinate.Enum( 126 - ii * 4))
            if ii < no_generators:
                self.route(bg_top, drv_top)
                self.route(bg_bottom, drv_bottom)
                self.enable_synapse_line(drv_top, l1address)
                self.enable_synapse_line(drv_bottom, l1address)
            else:
                self.disable_synapse_line(drv_top)
                self.disable_synapse_line(drv_bottom)


    def enable_synapse_line(self, driver_c, l1address, exitatory=True):
        """
        """

        left = Coordinate.left
        right = Coordinate.right
        top = Coordinate.top
        bottom = Coordinate.bottom

        driver = self.hicann.synapses[driver_c]
        driver_decoder = l1address.getDriverDecoderMask()
        driver.set_l1();
        driver[top].set_decoder(top, driver_decoder)
        driver[top].set_decoder(bottom, driver_decoder)
        driver[top].set_gmax_div(left, 1)
        driver[top].set_gmax_div(right, 1)
        driver[top].set_syn_in(left, 1) # Esynx
        driver[top].set_syn_in(right, 0) # Esyni, perhaps
        driver[top].set_gmax(0)
        driver[bottom] = driver[top]
        driver[bottom].set_syn_in(left, 0)
        driver[bottom].set_syn_in(right, 1)

        if exitatory:
            w_top = [pyhalbe.HICANN.SynapseWeight(15)] * 256
            w_bottom = [pyhalbe.HICANN.SynapseWeight(0)] * 256
        else:
            w_top = [pyhalbe.HICANN.SynapseWeight(0)] * 256
            w_bottom = [pyhalbe.HICANN.SynapseWeight(15)] * 256


        synapse_line_top    = Coordinate.SynapseRowOnHICANN(driver_c, top)
        synapse_line_bottom = Coordinate.SynapseRowOnHICANN(driver_c, bottom)
        self.hicann.synapses[synapse_line_top].weights[:] = w_top
        self.hicann.synapses[synapse_line_bottom].weights[:] = w_bottom
        synapse_decoder = [l1address.getSynapseDecoderMask()] * 256
        self.hicann.synapses[synapse_line_top].decoders[:] = synapse_decoder
        self.hicann.synapses[synapse_line_bottom].decoders[:] = synapse_decoder
        self.logger.DEBUG("enabled {!s} listing to {!s}".format(driver_c, l1address))

    def disable_synapse_line(self, driver_c):
        """
        """

        top = Coordinate.top
        bottom = Coordinate.bottom

        driver = self.hicann.synapses[driver_c]
        driver.disable();
        synapse_line_top    = Coordinate.SynapseRowOnHICANN(driver_c, top)
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
        repeater = output_buffer.repeater().horizontal()
        out_line = output_buffer.repeater().horizontal().line()
        driver_line = driver.line()
        repeater_data = self.hicann.repeater[repeater]
        repeater_data.setOutput(Coordinate.right, True)
        if driver.side() == Coordinate.left:
            v_line_value = 31 - out_line.value()/2
            v_line_value += 32 * route
        if driver.side() == Coordinate.right:
            v_line_value = 128 + out_line.value()/2
            v_line_value += 32 * route
        v_line = Coordinate.VLineOnHICANN(v_line_value)
        chain = (output_buffer, repeater, out_line, v_line, driver_line, driver)
        dbg = " -> ".join(['{!s}'] * len(chain))
        self.logger.DEBUG("connected " + dbg.format(*chain))
        self.hicann.crossbar_switches.set(v_line, out_line, True)
        self.hicann.synapse_switches.set(v_line, driver_line, True)


    def set_current_stimulus(self, stimulus):
        """Updates current stimulus for all neurons"""
        #for block in Coordinate.iter_all(Coordinate.FGBlockOnHICANN):
        for block in range(4):
            self.hicann.current_stimuli[block] = stimulus
        # TODO write to FG

    def switch_current_stimulus_and_output(self, coord_neuron):
        """ Switches the current stimulus and analog output to a certain neuron.
            To avoid invalid neuron configurations (see HICANN doc page 33),
            all aouts and current stimuli are disabled before enabling them for one neuron."""
        if not self._connected:
            self.connect()
        self.hicann.disable_aout()
        self.hicann.disable_current_stimulus()
        self.hicann.enable_current_stimulus(coord_neuron)
        self.hicann.enable_aout(coord_neuron, Coordinate.AnalogOnHICANN(0))
        self.wafer.configure(UpdateAnalogOutputConfigurator())

