"""Helper classes for StHAL."""

import math
import numpy
import pandas
import pyhalbe
import pysthal
import pylogging
from pyhalbe import HICANN
from pyhalbe.HICANN import neuron_parameter
from pyhalbe.HICANN import shared_parameter
from pyhalco_common import iter_all, Enum, X, Y, top, bottom, left, right
from pyhalco_hicann_v2 import AnalogOnHICANN
from pyhalco_hicann_v2 import BackgroundGeneratorOnHICANN
from pyhalco_hicann_v2 import FGBlockOnHICANN
from pyhalco_hicann_v2 import FGCellOnFGBlock
from pyhalco_hicann_v2 import FGRowOnFGBlock
from pyhalco_hicann_v2 import GbitLinkOnHICANN
from pyhalco_hicann_v2 import NeuronOnFGBlock
from pyhalco_hicann_v2 import SynapseDriverOnHICANN
from pyhalco_hicann_v2 import DNCMergerOnHICANN
from pyhalco_hicann_v2 import VLineOnHICANN
from pyhalco_hicann_v2 import SynapseRowOnHICANN
from pyhalco_hicann_v2 import FGRowOnFGBlock
from pyhalco_hicann_v2 import NeuronOnHICANN

from pysthal import HICANNConfigurator as HICANNv2Configurator
from pysthal import HICANNv4Configurator


class UpdateParameterUp(HICANNv4Configurator):
    """
    New HICANN configurator, could do stuff better or worse...

    Warning: It is assumend that the paramter is growing!
    """
    def __init__(self, parameters):
        HICANNv4Configurator.__init__(self)
        self.rows = [
            pyhalbe.HICANN.FGRowOnFGBlock4([
                HICANN.getNeuronRow(b, p) for b in iter_all(FGBlockOnHICANN)])
                    for p in parameters if isinstance(p, neuron_parameter)]
        self.rows += [
            pyhalbe.HICANN.FGRowOnFGBlock4([
                HICANN.getSharedRow(b, p) for b in iter_all(FGBlockOnHICANN)])
                    for p in parameters if isinstance(p, shared_parameter)]
        self.rows = self.row_list_t(self.rows)

    def config_fpga(self, fpga_handle, fpga):
        pass

    def config(self, fpga_handle, handle, hicann):
        row_lists = HICANNv4Configurator.row_lists_t()
        row_lists.append(self.rows)

        # cf. #3131
        hicann_handles = pysthal.vector_less__boost_scope_shared_ptr_less_HMF_scope_Handle_scope_HICANN_greater___greater_()
        hicann_handles.append(handle)

        hicann_datas = pysthal.vector_less__boost_scope_shared_ptr_less_sthal_scope_HICANNData_greater___greater_()
        hicann_datas.append(hicann)

        self.program_normal(hicann_handles, hicann_datas, row_lists)

class UpdateParameterUpAndConfigure(UpdateParameterUp):
    def config(self, fpga_handle, handle, hicann):
        UpdateParameterUp.config(self, fpga_handle, handle, hicann)

        self.config_neuron_quads(handle, hicann)
        self.config_phase(handle, hicann)
        self.config_gbitlink(handle, hicann)
        self.config_synapse_array(handle, hicann)
        self.config_synapse_drivers(handle, hicann)
        self.config_synapse_switch(handle, hicann)
        self.config_crossbar_switches(handle, hicann)
        self.config_repeater(handle, hicann)
        self.config_merger_tree(handle, hicann)
        self.config_dncmerger(handle, hicann)
        self.config_background_generators(handle, hicann)
        self.lock_repeater(handle, hicann)
        self.flush_hicann(handle)


class SpikeReadoutHICANNConfigurator(HICANNv2Configurator):

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
        self.flush_hicann(handle)
        self.lock_repeater(handle, data)

        self.config_neuron_config(handle, data)
        self.config_neuron_quads(handle, data)
        self.config_analog_readout(handle, data)
        self.flush_hicann(handle)


class UpdateAnalogOutputConfigurator(HICANNv2Configurator):
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
        self.flush_hicann(h)


class SetFGCell(HICANNv2Configurator):
    """Puts the given FG cell to the analog output
    """
    def __init__(self, nrn, parameter):
        HICANNv2Configurator.__init__(self)
        self.coords = (nrn, parameter)

    def config_fpga(self, *args):
        """do not reset FPGA"""
        pass

    def config(self, fpga_handle, h, hicann):
        """Call analog output related configuration functions."""
        HICANN.set_fg_cell(h, *self.coords)
        self.flush_hicann(h)


class UpdateParameter(HICANNv2Configurator):
    def __init__(self, neuron_parameters):
        HICANNv2Configurator.__init__(self)
        self.blocks = dict(
            (row, []) for row in iter_all(FGRowOnFGBlock))
        for parameter in neuron_parameters:
            for block in iter_all(FGBlockOnHICANN):
                self.blocks[HICANN.getNeuronRow(block, parameter)].append(block)

    def hicann_init(self, h):
        HICANN.init(h, False)

    def config_synapse_array(self, handle, data):
        pass

    def write_fg_row(self, handle, row, fg, write_down):
        blocks = self.blocks[row]
        if len(blocks) == 4:
            HICANNv2Configurator.write_fg_row(
                self, handle, row, fg, write_down)
        else:
            for block in blocks:
                data = fg.getBlock(block).getFGRow(row)
                self.getLogger().info("Update {} on {}".format(row, block))
                HICANN.set_fg_row_values(handle, block, row, data, write_down)


class ADCFreqConfigurator(HICANNv4Configurator):
    """
    Attention: Do not disable writing of floating gates otherwise
    resuming an aborted experiment or starting a calibration which
    needs the ADC frequency might not work as the floating gates might be in a
    state that prohibits proper operation.
    """

    def hicann_init(self, h):
        pyhalbe.HICANN.init(h, False)

    def config_synapse_array(self, handle, data):
        pass

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

    def traceRaw(self):
        pass

    def setRecordingTime(self, time):
        pass

    def getRecordingTime(self):
        pass

    def getTimestamp(self):
        pass

    def getTimestamps(self):
        pass

    def status(self):
        pass


class StHALContainer(object):
    """
    Contains StHAL objects for hardware access.
    Multiple experiments can share one container.

    Attributes:
        coord_analog: AnalogOnHICANN analog channel used for readout
        coord_hicann: HICANNOnWafer HICANN used for measurements
        coord_wafer: Wafer wafer used for measurements
        hicann_version: int setting the HICANN version
        recording_time: lenght of recorded traces
    """
    def __getstate__(self):
        odict = self.__dict__.copy()
        odict['adc'] = None
        odict['_connected'] = False
        return odict

    logger = pylogging.get("pycake.helper.sthal")

    def __init__(self,
                 config,
                 coord_analog,
                 recording_time=1.e-4):
        """Initialize StHAL. kwargs default to vertical setup configuration.

        Args:
            config: Config instance, used to determine Wafer, HICANN, PLL,
                    big_cap, wafer_cfg, speedup
                wafer_cfg: load wafer configuration from this file, ignore rest
            coord_analog: AnalogOnHICANN Coordinate
            recording_time: ADC recording time in seconds
            neuron_size: Number of denmems to connect
        """

        self.coord_wafer, self.coord_hicann = config.get_coordinates()
        self.hicann_version = config.get_hicann_version()
        self.wafer_cfg = config.get_wafer_cfg()
        self.save_raw_traces = config.get_save_raw_traces()

        # FIXME: should we store the content of the hwdb file instead?
        self.hwdb = config.get_hwdb()

        self.wafer = pysthal.Wafer(self.coord_wafer)
        if self.wafer_cfg:
            self.logger.info("Loading {}".format(self.wafer_cfg))
            self.wafer.load(self.wafer_cfg)
        else:
            self.setPLL(config.get_PLL())

        self.adc = None
        self.recording_time = recording_time
        self.coord_analog = coord_analog
        self._connected = False
        self.input_spikes = {}
        self.neuron_size = 1
        self.ideal_adc_freq = 96e6

    def __del__(self):
        if self._connected:
            self.disconnect()

    @property
    def hicann(self):
        return self.wafer[self.coord_hicann]

    @hicann.setter
    def hicann(self, hicann):
        self.wafer[self.coord_hicann] = hicann

    def connect(self):
        """Connect to the hardware."""
        if self.hwdb:
            self.logger.warn("using non-default hardware database {}".format(self.hwdb))
            db = pysthal.YAMLHardwareDatabase(self.hwdb)
        else:
            db = pysthal.MagicHardwareDatabase()
        self.wafer.connect(db)
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

        adc = self.hicann.analogRecorder(coord_analog)
        adc.setRecordingTime(self.recording_time)
        self.adc = adc

    def disconnect(self):
        """Free handles."""
        self.wafer.disconnect()
        self.adc.freeHandle()
        self.adc = None
        self._connected = False

    def write_config(self, configurator=None):
        """Write full configuration."""

        # cf. #3143
        settings = pysthal.Settings.get()
        settings.synapse_switches.max_switches_per_column_per_side = 2

        if not self._connected:
            self.connect()
        if configurator is None:
            if self.hicann_version == 2:
                self.logger.DEBUG("Using HICANNv2Configurator")
                configurator = HICANNv2Configurator()
            elif self.hicann_version == 4:
                self.logger.DEBUG("Using HICANNv4Configurator")
                configurator = HICANNv4Configurator()
            else:
                raise RuntimeError("No HICANNConfigurator assigned to given HICANN version: {}".format(self.hicann_version))

        for _ in range(3):
            try:
                return self.wafer.configure(configurator)
            except RuntimeError as err:
                if err.message == 'fg_log_error timeout':
                    self.logger.error("Error configuring floating gates, retry")
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

        # assume firing is enabled, this is the user's responsibility
        if not self.wafer_cfg and l1address is not None:
            self.hicann.enable_l1_output(coord_neuron, pyhalbe.HICANN.L1Address(l1address))
        self.hicann.enable_aout(coord_neuron, self.coord_analog)
        self.write_config(configurator=UpdateAnalogOutputConfigurator())

    def send_spikes_to_all_neurons(self, spike_times, excitatory=True,
                                   gmax_div=2, gmax=0, weight=15, locking_freq=0.05e6):
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

            self.hicann.route(link.toDNCMergerOnHICANN(), driver)
            self.enable_synapse_line(
                driver, stim_l1address, gmax_div=gmax_div, excitatory=excitatory,
                gmax=gmax, weight=weight)

            self.hicann.layer1[link] = HICANN.GbitLink.TO_HICANN

            self.hicann.sendSpikes(link, spikes)

    def read_adc(self, recording_time=None):
        """Run experiment, read out ADC.
        """
        if not self._connected:
            self.connect()
        max_tries = 10
        if recording_time is None:
            recording_time = self.recording_time
        for ii in range(max_tries):
            try:
                self.adc.record(recording_time)
                v = self.adc.trace()
                t = numpy.arange(len(v)) * self.adc.getTimestamp()
                df = pandas.DataFrame({'v': v}, index=t)
                if self.save_raw_traces:
                    df['v_raw'] = self.adc.traceRaw()
                return df
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
                # self._sendSpikes(self.input_spikes)
                runtime = self.adc.getRecordingTime() + 1.e-5
                runner = pysthal.ExperimentRunner(runtime)
                self.wafer.restart(runner)  # Clears received spikes
                self.adc.record()  # TODO triggered
                traces = pandas.DataFrame({'v': self.adc.trace()},
                                           index=self.adc.getTimestamps())

                recv_spikes = []
                for link in iter_all(GbitLinkOnHICANN):
                    tmp = self.hicann.receivedSpikes(link)
                    if len(tmp):
                        times, addrs = tmp.T
                        recv_spikes.append(pandas.DataFrame(
                            {'L1Address': addrs, 'GbitLink': int(link)},
                            index=times, dtype=numpy.int8))
                if recv_spikes:
                    return traces, pandas.concat(recv_spikes)
                else:
                    return traces, pandas.DataFrame(columns=['L1Address', 'GbitLink'],
                                                    dtype=numpy.int8)
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
            cfg.fg_biasn = biasn
            fg.setFGConfig(Enum(ii), cfg)

    def stimulateNeurons(self, rate, no_generators, excitatory=True,
                         gmax=0, gmax_div=2, weight=15):
        """Stimulate neurons via background generators

        Args:
            rate: Rate of a single generator in Hertz. This value will be
                  rounded upwards to the next possible value for the given
                  PLL setting.
            number: Number of generators to use in parallel (per Neuron)

        Returns:
            Actual spike rate in Hz
        """
        assert(no_generators >= 0 and no_generators <= 4)
        assert(rate <= 5.0e6)
        assert(1 <= gmax_div <= 30)
        assert(0 <= weight <= 15)
        assert(0 <= gmax <= 3)

        self.hicann.clear_complete_l1_routing()
        l1address = pyhalbe.HICANN.L1Address(0)

        PLL = self.getPLL()
        bg_period = int(math.floor(PLL/rate) - 1)
        rate = PLL/(bg_period + 1) # Actual spike rate
        self.logger.info("Stimulating neurons from {} background generators"
                         " with isi {}".format(no_generators, bg_period))

        for bg in iter_all(BackgroundGeneratorOnHICANN):
            generator = self.hicann.layer1[bg]
            generator.enable(bg.value()/2 < no_generators)
            generator.random(False)
            generator.period(bg_period)
            generator.address(l1address)
            self.logger.DEBUG("activate {!s} with period {}".format(bg, bg_period))

        links = []
        for ii in range(4):
            bg_top = DNCMergerOnHICANN(2*ii+1)
            bg_bot = DNCMergerOnHICANN(2*ii)
            links.append(GbitLinkOnHICANN(2 * ii + 1))
            links.append(GbitLinkOnHICANN(2 * ii))

            drv_top = SynapseDriverOnHICANN(
                Enum(83 + ii * 4))
            drv_bot = SynapseDriverOnHICANN(
                Enum(142 - ii * 4))
            if ii < no_generators:
                self.route(bg_top, drv_top)
                self.route(bg_bot, drv_bot)
                self.enable_synapse_line(
                    drv_top, l1address, gmax_div=gmax_div, excitatory=excitatory,
                    gmax=gmax, weight=weight)
                self.enable_synapse_line(
                    drv_bot, l1address, gmax_div=gmax_div, excitatory=excitatory,
                    gmax=gmax, weight=weight)
            else:
                self.disable_synapse_line(drv_top)
                self.disable_synapse_line(drv_bot)
        return rate

    def mirror_synapse_driver(self, driver, count, upwards=True, skip=0):
        """
        Connect a driver to its neighbour, the settings are copied from the
        driver.
        Works only for upper half

        Args:
            driver: [SynapseDriverOnHICANN] driver to mirror
            count: Number of drivers to add
            skip: Skip this drivers (unused? blacklist?)
            upwards: direction (ignored, but would be nice)
        """
        if skip > 0 or not upwards:
            raise NotImplementedError("skipping drivers is not yet implemeted")

        self.hicann.synapses[driver].connect_neighbor = True
        for ii in xrange(count):
            target_driver = SynapseDriverOnHICANN(driver.x(), Y(int(driver.y()) - 2))
            self.hicann.synapses[target_driver] = self.hicann.synapses[driver]
            self.hicann.synapses[target_driver].locin = False

            for ii in (top, bottom):
                synapse_line = SynapseRowOnHICANN(driver, ii)
                target_synapse_line = SynapseRowOnHICANN(target_driver, ii)
                self.hicann.synapses[target_synapse_line].decoders = self.hicann.synapses[synapse_line].decoders
                self.hicann.synapses[target_synapse_line].weights = self.hicann.synapses[synapse_line].weights
            driver = target_driver

    def stimulatePreout(self, rate, aout):
        """
        Stimulate the synapse drivers, which have the debug output. And connects
        the debug output to the analog output.

        Returns:
            Actual spike rate in Hz
        """
        assert(rate <= 5.0e6)
        self.hicann.clear_complete_l1_routing()

        l1address = pyhalbe.HICANN.L1Address(0)
        if self.hicann_version == 2:
            drivers = (SynapseDriverOnHICANN(Enum(111)),
                       SynapseDriverOnHICANN(Enum(112)))
            bg = BackgroundGeneratorOnHICANN(7)
        elif self.hicann_version == 4:
            drivers = (SynapseDriverOnHICANN(Enum(109)),
                       SynapseDriverOnHICANN(Enum(114)))
            bg = BackgroundGeneratorOnHICANN(6)
        else:
            raise RuntimeError("Invalid HICANN version")

        PLL = self.getPLL()
        bg_period = int(math.floor(PLL/rate) - 1)
        rate = PLL / (bg_period + 1)
        self.logger.info("Stimulating preout from {} with isi {}".format(
            bg, bg_period))

        generator = self.hicann.layer1[bg]
        generator.enable(True)
        generator.random(False)
        generator.period(bg_period)
        generator.address(l1address)
        self.logger.DEBUG("activate {!s} with period {}".format(bg, bg_period))

        for drv in drivers:
            self.route(DNCMergerOnHICANN(bg.value()), drv)
            self.configure_synapse_driver(drv, l1address, gmax_div=2, gmax=0)

        self.hicann.analog.set_preout(aout)

        return rate 

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
        assert(0 <= gmax <= 3)
        assert(1 <= gmax_div <= 30)

        from pyhalco_common import left, right, top, bottom

        driver = self.hicann.synapses[driver_c]
        driver_decoder = l1address.getDriverDecoderMask()
        driver.set_l1()
        driver[top].set_decoder(top, driver_decoder)
        driver[top].set_decoder(bottom, driver_decoder)
        driver[top].set_gmax(gmax)
        driver[top].set_gmax_div(HICANN.GmaxDiv(gmax_div))

        # copy config above
        driver[bottom] = driver[top]

        # Configure top to exictatory inputs and bottom to inhibitory
        driver[top].set_syn_in(left, 1)  # Esynx
        driver[top].set_syn_in(right, 0)  # Esyni, perhaps
        driver[bottom].set_syn_in(left, 0)
        driver[bottom].set_syn_in(right, 1)

    def enable_synapse_line(self, driver_c, l1address, gmax_div=2,
                            excitatory=True, weight=15, gmax=0):
        """
        """
        from pyhalco_common import top, bottom
        assert(0 <= weight <= 15)

        self.configure_synapse_driver(driver_c, l1address, gmax_div, gmax)

        if excitatory:
            w_top = [pyhalbe.HICANN.SynapseWeight(weight)] * 256
            w_bottom = [pyhalbe.HICANN.SynapseWeight(0)] * 256
        else:
            w_top = [pyhalbe.HICANN.SynapseWeight(0)] * 256
            w_bottom = [pyhalbe.HICANN.SynapseWeight(weight)] * 256

        synapse_line_top = SynapseRowOnHICANN(driver_c, top)
        synapse_line_bottom = SynapseRowOnHICANN(driver_c, bottom)
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

        top = top
        bottom = bottom

        driver = self.hicann.synapses[driver_c]
        driver.disable()
        synapse_line_top = SynapseRowOnHICANN(driver_c, top)
        synapse_line_bottom = SynapseRowOnHICANN(driver_c, bottom)
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
        repeater_data.setOutput(right, True)
        if driver.toSideHorizontal() == left:
            v_line_value = 31 - out_line.value()/2
            v_line_value += 32 * route
        if driver.toSideHorizontal() == right:
            v_line_value = 128 + out_line.value()/2
            v_line_value += 32 * route
        v_line = VLineOnHICANN(v_line_value)
        chain = (output_buffer, repeater, out_line, v_line, driver_line, driver)
        dbg = " -> ".join(['{!s}'] * len(chain))
        self.logger.DEBUG("connected " + dbg.format(*chain))
        self.hicann.crossbar_switches.set(v_line, out_line, True)
        self.hicann.synapse_switches.set(v_line, driver_line, True)

    def set_current_stimulus(self, stim_value, stim_length, pulse_length=15):
        """ Set current stimulus for all 4 blocks. Does not write to hardware.

            The stimulus is continuously active for stim_length parts with individual
            pulse_length, inactive for 129-stim_length parts.

            Args:
                stim_value: int -> strength of current (DAC -> nA conversion
                            unclear, but ~ 2nA per 500 DAC)
                stim_length: int -> how long should the current be?
                             max. duration is a whole cycle: 129

            Returns:
                length of one pulse in seconds (hardware time)
        """
        stimulus = pysthal.FGStimulus()
        stimulus.setPulselength(pulse_length)
        stimulus.setContinuous(True)

        stimulus[:stim_length] = [stim_value] * stim_length
        stimulus[stim_length:] = [0] * (len(stimulus) - stim_length)

        for block in range(4):
            self.hicann.current_stimuli[block] = stimulus

        # actual pulse length on chip is (pulse_length+1) / slow clock for each FG memory entry
        return (pulse_length+1) * stim_length / self.getSlowClock()

    def get_current_stimulus(self):
        """Assumes stimulus was created using set_current_stimulus above,
        resulting in a rectangular stimulus with a single amplitude and only
        one rectangle per period."""

        params = []
        for block in range(4):
            stimulus = self.hicann.current_stimuli[block]
            pulse_length = stimulus.getPulselength()
            assert(stimulus.getContinuous())
            stim_length = 129-len([s for s in stimulus if s==0])
            stim_value = stimulus[0]
            assert(stim_value > 0)
            for i in range(stim_length):
                assert(stimulus[i] == stim_value)
            params.append({"stim_length": stim_length, "pulse_length": pulse_length, "stim_value": stim_value})

        for p in params:
            assert(p == params[0])

        return params[0]

    def getSlowClock(self):
        """In the specification document this clock is called 'slow HICANN clock'"""
        return self.getPLL()/4.0

    def switch_current_stimulus_and_output(self, coord_neuron, l1address=None):
        """ Switches the current stimulus and analog output to a certain neuron.
            To avoid invalid neuron configurations (see HICANN doc page 33),
            all aouts and current stimuli are disabled before enabling them for
            one neuron. Firing of all neuron is also deactivated. Only the firing
            mechanism of the chosen neuron is activated if l1address is not None."""
        if not self._connected:
            self.connect()
        # disable everything except firing
        self.hicann.disable_aout()
        self.hicann.disable_current_stimulus()
        self.hicann.disable_l1_output()

        # now enable this neuron
        self.hicann.enable_current_stimulus(coord_neuron)
        if not l1address is None:
            self.hicann.enable_l1_output(coord_neuron, pyhalbe.HICANN.L1Address(l1address))
        self.hicann.enable_aout(coord_neuron, self.coord_analog)
        self.wafer.configure(UpdateAnalogOutputConfigurator())

    def get_pulse_length(self, coord_neuron):
        """Get the length of the currently configured current stimulus.
        """
        raise NotImplemented
        if coord_neuron not in self.get_stimulated_neurons():
            return None

        return self.get_current_stimulus()['pulse_length']

    def get_stimulated_neurons(self):
        """Which neurons are connected to stimulus"""
        h = self.hicann
        return [nrn for nrn in iter_all(NeuronOnHICANN) if h.neurons[nrn].enable_current_input()]

    def set_recording_time(self, recording_time, repetitions):
        """Sets the recording time of the ADC.

        The recording_time should be as small as theoretical required for the
        measurement. And the repetitions factor should be the amount you need
        to cope with readout noise. This is to speed up simulations
        """

        self.recording_time = recording_time * repetitions

    def set_neuron_size(self, n):
        self.logger.INFO("Setting neuron size to {}".format(n))
        self.firing_denmems = list(self.hicann.set_neuron_size(n))
        self.neuron_size = n

    def get_neuron_size(self):
        return self.neuron_size

    def get_speedup_type(self, speedup_str):
        """
        speedup: "slow", "normal" or "fast" (case-insensitive)
        returns sthal speedup type
        """

        speedup_type = pysthal.SpeedUp.names[speedup_str.upper()]
        return speedup_type

    def set_speedup(self, speedup):
        """ sets speedup for I_gl, I_gladapt and I_radapt
        """
        self.logger.INFO("Setting speedup to {}".format(pysthal.SpeedUp.values[speedup]))
        self.hicann.set_fg_speed_up_scaling(speedup)

    def get_speedup(self):
        """ get speedup for I_gl, I_gladapt and I_radapt if all are equal
        """
        s_gl = self.hicann.get_speed_up_gl()
        s_gladapt = self.hicann.get_speed_up_gladapt()
        s_radapt = self.hicann.get_speed_up_radapt()
        errmsg = "SpeedUps not equal. gl: {}, gladapt: {}, radapt: {}".format(s_gl, s_gladapt, s_radapt)
        assert(s_gl == s_gladapt == s_radapt), errmsg
        return s_gl

    def set_bigcap(self, bigcap):
        s = {True: "big", False: "small"}
        self.logger.INFO("Using {} capacitors.".format(s[bigcap]))
        self.hicann.use_big_capacitors(bigcap)

    def get_bigcap(self):
        return self.hicann.get_bigcap_setting()

    def read_all_floating_gates(self):
        """
        Helper method to read back all floating gate voltages.

        Note: This is currently not integrated, but can just be hacked into
        ADCMeasurement._measure if needed.

        Returns:
            pandas.DataFrame containing the raw voltages read from hardware
        """
        from pyhalbe.HICANN import getNeuronParameter, getSharedParameter
        from pysthal import ReadFloatingGates

        def get_row_names(block, row):
            try:
                shared = getSharedParameter(block, row).name
            except IndexError:
                shared = 'n.c.'
            try:
                neuron = getNeuronParameter(block, row).name
            except IndexError:
                neuron = 'n.c.'
            return (shared, neuron, int(row))

        self.adc.freeHandle()
        self.adc = None
        cfg = ReadFloatingGates(False)
        self.write_config(configurator=cfg)

        fg_values = {}
        for block in iter_all(FGBlockOnHICANN):
            tmp = numpy.empty((FGCellOnFGBlock.y_type.size, FGCellOnFGBlock.x_type.size))
            for cell in iter_all(FGCellOnFGBlock):
                tmp[cell.y()][cell.x()] = cfg.getMean(block, cell)

            index = pandas.MultiIndex.from_tuples(
                [get_row_names(block, row) for row in iter_all(FGRowOnFGBlock)],
                names=['shared', 'neuron', 'row'])
            fg_values[int(block.toEnum())] = pandas.DataFrame(tmp, index=index)
        result = pandas.concat(fg_values, names=['block'])
        self.connect_adc()
        return result

    def read_floating_gates(self, parameter):
        """
        Read floating gate voltages for the given parameter.
        Only the floating gates blocks accessabel from the used analog port are
        read.

        Note: This is currently not integrated, but can just be hacked into
        ADCMeasurement._measure if needed.

        Returns:
            pandas.DataFrame containing the raw voltages read from hardware
        """


        fgblocks = [
            FGBlockOnHICANN(X(0), Y(self.coord_analog.value())),
            FGBlockOnHICANN(X(1), Y(self.coord_analog.value())),
        ]

        result = {}
        for fgblock in fgblocks:
            if fgblock.x() == X(0):
                self.hicann.analog.set_fg_left(self.coord_analog)
            else:
                self.hicann.analog.set_fg_right(self.coord_analog)

            self.write_config(configurator=UpdateAnalogOutputConfigurator())
            data = []
            for blk_nrn in iter_all(NeuronOnFGBlock):
                nrn = blk_nrn.toNeuronOnHICANN(fgblock)
                self.write_config(configurator=SetFGCell(nrn, parameter))
                trace = self.read_adc(1.0e-4)
                data.append((nrn, trace['v'].mean(), trace['v'].std()))
            result[fgblock] = pandas.DataFrame(
                data, columns=['neuron', 'mean', 'variance'])
            result[fgblock].set_index('neuron', inplace=True)
        return pandas.concat(result)

    def is_hardware(self):
        return True
