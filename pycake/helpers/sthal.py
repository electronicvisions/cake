"""Helper classes for StHAL."""

import math
import os
import hashlib
import cPickle
import numpy
import pandas
import pyhalbe
import pysthal
import pylogging
import scipy.interpolate
import Coordinate
from pyhalbe import HICANN
from Coordinate import iter_all
from Coordinate import Enum, X, Y
from Coordinate import AnalogOnHICANN
from Coordinate import BackgroundGeneratorOnHICANN
from Coordinate import FGBlockOnHICANN
from Coordinate import FGCellOnFGBlock
from Coordinate import FGRowOnFGBlock
from Coordinate import GbitLinkOnHICANN
from Coordinate import NeuronOnFGBlock
from Coordinate import NeuronOnHICANN
from Coordinate import SynapseDriverOnHICANN
from collections import defaultdict
from sims.sim_denmem_lib import NETS_AND_PINS
from sims.sim_denmem_lib import TBParameters
from sims.sim_denmem_lib import run_remote_simulation


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


class HICANNConfigurator(pysthal.HICANNConfigurator):
    """
    New HICANN configurator, could do stuff better or worse...
    """

    FG_SHAPE = (FGCellOnFGBlock.y_type.size, FGCellOnFGBlock.x_type.size)
    ALL_ROWS = tuple(row for row in iter_all(FGRowOnFGBlock))
    V_ROWS = [[row]*4 for row in ALL_ROWS[::2]]
    C_ROWS = [[row]*4 for row in ALL_ROWS[1::2]]
    CURRENT_PARAMETERS = [
        p for n, p in pyhalbe.HICANN.neuron_parameter.names.iteritems()
        if n[0] == 'I']

    def __init__(self, readout_block=None):
        pysthal.HICANNConfigurator.__init__(self)
        self.readout = None
        self.readout_block = readout_block
        self.readout_values = []
        if readout_block is not None:
            self.readout = pysthal.ReadFloatingGates(
                do_reset=False, do_read_default=False)
            for cell in iter_all(FGCellOnFGBlock):
                self.readout.setReadCell(readout_block, cell, True)

    def config(self, fpga_handle, handle, hicann):
        # store the handle for readout, a bit hacky ...
        self.fpga_handle = fpga_handle
        pysthal.HICANNConfigurator.config(self, fpga_handle, handle, hicann)
        del self.fpga_handle

    def hicann_init(self, h):
        HICANN.init(h, False)

    def config_floating_gates(self, handle, hicann):
        fpga_handle = self.fpga_handle
        low, high = self.get_current_rows(hicann)
        self.zero_fg(handle, hicann)
        self.programm_normal(handle, hicann, self.V_ROWS + low)
        self.programm_high(handle, hicann, high)

    @staticmethod
    def make_df(data, block):
        assert data.shape == FG_SHAPE
        index = pandas.MultiIndex.from_tuples(
                [get_row_names(block, row) for row in iter_all(FGRowOnFGBlock)],
                names=['shared', 'neuron', 'row'])
        return pandas.DataFrame(data, index=index)

    def fgblock_to_array(self, hicann):
        fgblock = hicann.floating_gates[self.readout_block]
        values = numpy.empty(self.FG_SHAPE, dtype=numpy.int16)
        for cell in iter_all(FGCellOnFGBlock):
            values[cell.y()][cell.x()] = fgblock.getRaw(cell)
        return values

    def read_fg(self, fpga_handle, handle, hicann, values):
        assert values.shape == self.FG_SHAPE
        block = self.readout_block
        values = values.astype(numpy.int16, copy=False)

        if self.readout is None:
            return

        self.readout.config(fpga_handle, handle, hicann)
        tmp = numpy.empty(FG_SHAPE)
        for cell in iter_all(FGCellOnFGBlock):
            tmp[cell.y()][cell.x()] = self.readout.getMean(block, cell)
        result = make_df(tmp, block), self.make_df(values, block)
        self.readout_values.append(result)

    def get_current_rows(self, hicann):
        low = []
        high = []
        fgc = hicann.floating_gates
        for param in self.CURRENT_PARAMETERS:
            is_high = True
            for block in iter_all(FGBlockOnHICANN):
                row= fgc[block].getFGRow(HICANN.getNeuronRow(block, param))
                data = numpy.array(
                    [row.getShared()] +
                    [row.getNeuron(nrn) for nrn in iter_all(NeuronOnFGBlock)])
                if numpy.any(data[1:] < 800):
                    is_high = False
            if is_high:
                high.append(param)
            else:
                low.append(param)
        low = [tuple(HICANN.getNeuronRow(b, p) for b in iter_all(FGBlockOnHICANN))
               for p in low]
        high = [tuple(HICANN.getNeuronRow(b, p) for b in iter_all(FGBlockOnHICANN))
                for p in high]
        return low, high

    def zero_fg(self, handle, hicann):
        """
        Writes floating gates fast to zero values

        First current cells than voltage cells.
        """
        fgconfig = hicann.floating_gates.getFGConfig(Enum(0))
        fgconfig.voltagewritetime = 63
        fgconfig.currentwritetime = 63
        fgconfig.maxcycle = 15
        fgconfig.pulselength = 15
        fgconfig.readtime = 30
        fgconfig.acceleratorstep = 2
        fgconfig.writeDown = fgconfig.WRITE_DOWN

        for block in iter_all(FGBlockOnHICANN):
            pyhalbe.HICANN.set_fg_config(handle, block, fgconfig)

        writeDown = fgconfig.WRITE_DOWN
        row_data = [pyhalbe.HICANN.FGRow()] * 4
        for row in self.C_ROWS + self.V_ROWS:
            pyhalbe.HICANN.set_fg_row_values(handle, row, row_data, writeDown)

    def programm_high(self, handle, hicann, rows):
        # Zero fgs
        fgconfig = hicann.floating_gates.getFGConfig(Enum(0))
        fgconfig.voltagewritetime = 63
        fgconfig.currentwritetime = 63
        fgconfig.maxcycle = 31
        fgconfig.pulselength = 15
        fgconfig.readtime = 30
        fgconfig.acceleratorstep = 2
        fgconfig.writeDown = fgconfig.WRITE_UP
        for block in iter_all(FGBlockOnHICANN):
            pyhalbe.HICANN.set_fg_config(handle, block, fgconfig)

        fgc = hicann.floating_gates
        for row in rows:
            data = [fgc[blk].getFGRow(row[int(blk.id())])
                    for blk in iter_all(FGBlockOnHICANN)]
            pyhalbe.HICANN.set_fg_row_values(
                handle, row, data, fgconfig.writeDown)

    def programm_normal(self, handle, hicann, rows):
        fgc = hicann.floating_gates
        for step in (1, 2):
            fgconfig = hicann.floating_gates.getFGConfig(Enum(step))

            for block in iter_all(FGBlockOnHICANN):
                pyhalbe.HICANN.set_fg_config(handle, block, fgconfig)

            for row in rows:
                data = [fgc[blk].getFGRow(row[int(blk.id())])
                        for blk in iter_all(FGBlockOnHICANN)]
                pyhalbe.HICANN.set_fg_row_values(
                    handle, row, data, fgconfig.writeDown)


class UpdateParameterUp(HICANNConfigurator):
    """
    New HICANN configurator, could do stuff better or worse...

    Warning: It is assumend that the paramter is growing!
    """
    def __init__(self, parameters, readout_block=None):
        HICANNConfigurator.__init__(self, Coordinate.FGBlockOnHICANN())
        self.rows = [
            tuple(HICANN.getNeuronRow(b, parameter)
                  for b in iter_all(FGBlockOnHICANN))
            for parameter in parameters]

    def config_fpga(self, fpga_handle, fpga):
        pass

    def config(self, fpga_handle, handle, hicann):
        for rows in self.rows:
            self.programm_normal(handle, hicann, self.rows)


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


class SetFGCell(pysthal.HICANNConfigurator):
    """Puts the given FG cell to the analog output
    """
    def __init__(self, nrn, parameter):
        pysthal.HICANNConfigurator.__init__(self)
        self.coords = (nrn, parameter)

    def config_fpga(self, *args):
        """do not reset FPGA"""
        pass

    def config(self, fpga_handle, h, hicann):
        """Call analog output related configuration functions."""
        HICANN.set_fg_cell(h, *self.coords)


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


class ADCFreqConfigurator(pysthal.HICANNConfigurator):
    def hicann_init(self, h):
        pyhalbe.HICANN.init(h, False)

    def config_synapse_array(self, handle, data):
        pass

    def config_floating_gates(self, handle, data):
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
        odict['adc'] = None
        odict['_connected'] = False
        return odict

    logger = pylogging.get("pycake.helper.sthal")

    def __init__(self,
                 config,
                 coord_analog=Coordinate.AnalogOnHICANN(0),
                 recording_time=1.e-4):
        """Initialize StHAL. kwargs default to vertical setup configuration.

        Args:
            config: Config instance, used to determine Wafer, HICANN, PLL,
                    big_cap, dump_file, wafer_cfg, speedup, fg_biasn
                wafer_cfg: load wafer configuration from this file, ignore rest
                dump_file: filename for StHAL dump handle instead of hardware
            coord_analog: AnalogOnHICANN Coordinate
            recording_time: ADC recording time in seconds
        """

        self.coord_wafer, self.coord_hicann = config.get_coordinates()
        self.wafer_cfg = config.get_wafer_cfg()
        self.dump_file = None  # TODO add to config
        self.wafer_cfg = config.get_wafer_cfg()

        self.wafer = pysthal.Wafer(self.coord_wafer)
        if self.wafer_cfg:
            self.logger.info("Loading {}".format(wafer_cfg))
            self.wafer.load(wafer_cfg)
        else:
            self.setPLL(config.get_PLL())
            self.set_bigcap(config.get_bigcap())
            # Get the SpeedUp type from string
            speedup = pysthal.SpeedUp.names[config.get_speedup().upper()]
            self.set_speedup(speedup)
            self.set_fg_biasn(config.get_fg_biasn())

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

    def write_config(self, configurator=None):
        """Write full configuration."""
        if not self._connected:
            self.connect()
        if configurator is None:
            configurator = HICANNConfigurator()

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

    def switch_analog_output(self, coord_neuron, enable_firing=True, l1address=None):
        """Write analog output configuration (only).
           If l1address is None, l1 output is disabled.

           Using write_config() after this function will result
           in errors like
           HostALController: FPGA expected sequence number 00000000 instead of the current 000xxxxx. Cannot correct this.
           HostALController::sendSingleFrame: did not get an answer from FPGA.
        """
        self.hicann.disable_aout()
        self.hicann.disable_l1_output()
        self.hicann.disable_firing()
        self.hicann.disable_current_stimulus()
        if enable_firing:
            self.hicann.enable_firing(coord_neuron)
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
                return pandas.DataFrame({'v': v}, index=t)
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
                self.wafer.restart(runner)  # Clears received spikes
                self.adc.record()  # TODO triggered
                traces = pandas.DataFrame({'v': self.adc.trace()},
                                           index=self.adc.getTimestamps())

                recv_spikes = []
                for link in Coordinate.iter_all(GbitLinkOnHICANN):
                    times, addrs = self.hicann.receivedSpikes(link)
                    recv_spikes.append(pandas.DataFrame(
                        {'L1Address': addr, 'GbitLink': link},
                        index=times, dtype=numpy.int8))

                return traces, pandas.concat(data)
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

    def stimulateNeurons(self, rate, no_generators, excitatory=True):
        """Stimulate neurons via background generators

        Args:
            rate: Rate of a single generator in Hertz
            number: Number of generators to use in parallel (per Neuron)
        """
        assert(no_generators >= 0 and no_generators <= 4)
        assert(rate <= 5.0e6)

        self.hicann.clear_complete_l1_routing()
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
                Coordinate.Enum(83 + ii * 4))
            drv_bottom = Coordinate.SynapseDriverOnHICANN(
                Coordinate.Enum(142 - ii * 4))
            if ii < no_generators:
                self.route(bg_top, drv_top)
                self.route(bg_bottom, drv_bottom)
                self.enable_synapse_line(drv_top, l1address, 2, excitatory)
                self.enable_synapse_line(drv_bottom, l1address, 2, excitatory)
            else:
                self.disable_synapse_line(drv_top)
                self.disable_synapse_line(drv_bottom)
        return links

    def stimulatePreout(self, rate):
        """
        Stimulate the synapse drivers, which have the debug output. And connects
        the debug output to the analog output.

        Note:
            This function will only work on HICANNv4, because the synapse drivers
            with debug outputs differ (v2: 111 and 112; v4: 109 and 114).

        Returns:
            outputbuffer used for stimulus
        """
        assert(rate <= 5.0e6)
        self.hicann.clear_complete_l1_routing()

        l1address = pyhalbe.HICANN.L1Address(0)
        gmax_div = 2
        drivers = (SynapseDriverOnHICANN(Enum(109)),
                   SynapseDriverOnHICANN(Enum(114)))
        bg = Coordinate.BackgroundGeneratorOnHICANN(6)
        output = bg.toOutputBufferOnHICANN()

        PLL = self.getPLL()
        bg_period = int(math.floor(PLL/rate) - 1)
        self.logger.info("Stimulating preout from {} with isi {}".format(
            bg, bg_period))

        generator = self.hicann.layer1[bg]
        generator.enable(True)
        generator.random(False)
        generator.period(bg_period)
        generator.address(l1address)
        self.logger.DEBUG("activate {!s} with period {}".format(bg, bg_period))

        for drv in drivers:
            self.route(output, drv)
            self.configure_synapse_driver(drv, l1address, gmax_div, 0)

        for analog in iter_all(AnalogOnHICANN):
            self.hicann.analog.set_preout(analog)

        return output

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

    def switch_current_stimulus_and_output(self, coord_neuron, enable_firing=True, l1address=None):
        """ Switches the current stimulus and analog output to a certain neuron.
            To avoid invalid neuron configurations (see HICANN doc page 33),
            all aouts and current stimuli are disabled before enabling them for
            one neuron. Firing of all neuron is also deactivated. Only the firing
            mechanism of the chosen neuron is activated if l1address is not None."""
        if not self._connected:
            self.connect()
        # disable everything
        self.hicann.disable_aout()
        self.hicann.disable_firing()
        self.hicann.disable_current_stimulus()
        self.hicann.disable_l1_output()
        # now enable this neuron
        self.hicann.enable_current_stimulus(coord_neuron)
        if enable_firing:
            self.hicann.enable_firing(coord_neuron)
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
        self.neuron_size = n

    def get_neuron_size(self):
        return self.neuron_size

    def set_speedup(self, speedup):
        self.logger.INFO("Setting speedup to {}".format(pysthal.SpeedUp.values[speedup]))
        self.hicann.set_fg_speed_up_scaling(speedup)

    def get_speedup(self):
        s_gl = self.hicann.get_speed_up_gl()
        s_gladapt = self.hicann.get_speed_up_gladapt()
        s_radapt = self.hicann.get_speed_up_radapt()
        errmsg = "SpeedUps not equal. gl: {}, gladapt: {}, radapt: {}".format(s_gl, s_gladapt, s_radapt)
        assert (s_gl == s_gladapt == s_radapt), errmsg
        return s_gl

    def set_bigcap(self, bigcap):
        s = {True: "big", False: "small"}
        self.logger.INFO("Using {} capacitors.".format(s[bigcap]))
        self.hicann.use_big_capacitors(bigcap)

    def get_bigcap(self):
        return self.hicann.get_bigcap_setting()

    def read_floating_gates(self):
        """
        Read back floating gate values from hardware.

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
            fg_values[int(block.id())] = pandas.DataFrame(tmp, index=index)
        result = pandas.concat(fg_values, names=['block'])
        self.connect_adc()
        return result

    def read_floating_gates(self, parameter):


        fgblocks = [
            FGBlockOnHICANN(X(0), Y(self.coord_analog.value())),
            FGBlockOnHICANN(X(1), Y(self.coord_analog.value())),
        ]

        result = {}
        for fgblock in fgblocks:
            if fgblock.x() == X(0):
                self.hicann.analog.set_fg_left(self.coord_analog)
            else:
                self.hicann.analog.set_fg_left(self.coord_analog)

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

    def __init__(self,
                 config,
                 coord_analog=Coordinate.AnalogOnHICANN(0),
                 recording_time=30.0e-6,
                 resample_output=True):
        """Initialize StHAL. kwargs default to vertical setup configuration.

        Args:
            config: Config instance
            recording_time: ADC recording time in seconds
            wafer_cfg: ?
            resample_output: bool
                if True, resample the result of read_adc to the adc
                sampling frequency
        """
        super(SimStHALContainer, self).__init__(
            config, coord_analog, recording_time)
        self.current_neuron = Coordinate.NeuronOnHICANN()
        host, port = config.get_sim_denmem().split(':')
        self.remote_host = host
        self.remote_port = int(port)

        self.resample_output = resample_output

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
        self.adc = FakeAnalogRecorder()

        maximum_spikes = config.get_sim_denmem_maximum_spikes()
        if maximum_spikes is None:
            self.maximum_spikes = 200
        else:
            self.maximum_spikes = maximum_spikes

        self.spike_counter_offset = self.simulation_init_time

        self.simulated_configurations = defaultdict(list)

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
        pass

        self.adc = None

    def write_config(self, configurator=None):
        """Write full configuration."""
        pass

    def switch_analog_output(self, coord_neuron, enable_firing=True, l1address=None):
        super(SimStHALContainer, self).switch_analog_output(
            coord_neuron, enable_firing, l1address)
        self.current_neuron = coord_neuron

    def resample_simulation_result(self, adc_result,
                                   adc_sampling_interval):
        """Re-sample the result of self.read_adc().

        adc_result: Dictionary as returned by self.read_adc(). Must
            contain the key "t". All values must be numpy arrays of
            same length.

        adc_sampling_interval: float. adc sampling interval in
            seconds.

        Returns a dictionary with the same keys as `adc_result`.
            self.resample_simulation_result(data)['t'] covers the
            linear range between min(data['t']) and max(data['t']) in
            steps corresponding to the adc sampling interval.

            For x != 't', self.resample_simulation_result(data)[x] is
            the interpolated value of data[x] at the corresponding
            time values.
        """
        time = numpy.arange(min(adc_result['t']), max(adc_result['t']),
                            adc_sampling_interval)

        if len(adc_result['t']) < 2:
            raise ValueError("simulated ADC output too short")

        self.logger.info(
            "resampling simulation adc_result from {} to {} samples".format(
                len(adc_result['t']), len(time)))
        self.logger.info(
            "new sampling interval is {} s".format(
                adc_sampling_interval))

        resampled = dict()
        signals = adc_result.keys()
        signals.remove('t')

        for signal in signals:
            resampled[signal] = scipy.interpolate.interp1d(
                adc_result['t'],
                adc_result[signal])(time)

        return pandas.DataFrame(resampled, index=time)

    def read_adc(self):
        """Fake ADC readout by evaluating a denmem_sim run."""

        param = self.build_TBParameters(self.current_neuron)
        json = param.to_json()
        result = self.run_simulation(self.current_neuron, param)

        if self.resample_output:
            return self.resample_simulation_result(
                result, adc_sampling_interval=1.0/self.ideal_adc_freq)
        else:
            return result

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

        if self.mc_seed is not None:
            mc_run = int(neuron.id())/2 + 1
            param.simulator_settings.set_mc_run(self.mc_seed, mc_run)

        # set the unused neuron to "harmless" parameters
        idx = (int(neuron.x()) + 1) % 2
        nparams = [
            ('El', 0.7),
            ('Vt', 1.2),
            ('Igl', 2.0e-6),
            ('Vreset', 0.7),
            ('Vconvoffi', 1.7),
            ('Vconvoffx', 1.7)
        ]
        fg = param.floating_gate_parameters
        for p, v in nparams:
            tmp = list(fg[p])
            tmp[idx] = v
            fg[p] = tuple(tmp)

        tmp = list(param.digital_parameters['activate_firing'])
        tmp[idx] = False
        param.digital_parameters['activate_firing'] = tuple(tmp)

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
            self.logger.info("Run simulation on {}:{}".format(
                self.remote_host, self.remote_port))
            lresult, rresult = run_remote_simulation(
                param, self.remote_host, self.remote_port,
                init_time=self.simulation_init_time)

            if json_hash:
                self.logger.info("cache result in {}".format(json_hash))
                with open(json_hash, 'w') as outfile:
                    data = (json, lresult, rresult)
                    cPickle.dump(data, outfile, cPickle.HIGHEST_PROTOCOL)

        if neuron == left:
            self.simulated_configurations[left].append(param)
            return lresult
        else:
            self.simulated_configurations[right].append(param)
            return rresult

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

    def get_neuron_size(self):
        return 1

    def switch_current_stimulus_and_output(self, coord_neuron, enable_firing=True, l1address=None):
        def do_nothing(*args, **kwargs):
            pass
        self.wafer.configure = do_nothing
        super(SimStHALContainer, self).switch_current_stimulus_and_output(coord_neuron, enable_firing, l1address)

        # remove non-C++ entry for pickling
        del self.wafer.__dict__['configure']

    def read_floating_gates(self, parameter):
        return pandas.DataFrame()

    def is_hardware(self):
        return False
