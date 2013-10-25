"""Helper classes for StHAL."""

import pyhalbe
import pysthal


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


class WriteFGTwiceConfigurator(pysthal.HICANNConfigurator):
    """Same as default configurator, but writes floating gates twice."""
    def config_fpga(self, f, fpga):
        """Copy of HICANNConfigurator::config_fpga()"""
        pyhalbe.FPGA.reset(f)

        self.config_dnc_link(f, fpga)

    def config(self, f, h, hicann):
        """Copy of HICANNConfigurator::config(), but writing floating gates twice."""

        self.reset_hicann(h, hicann)
        pyhalbe.HICANN.init(h)

        self.config_floating_gates(h, hicann)
        self.flush_fpga(f)
        self.config_floating_gates(h, hicann)
        self.config_fg_stimulus(h, hicann)

        self.config_neuron_quads(h, hicann)
        self.config_phase(h, hicann)
        self.config_gbitlink(h, hicann)

        self.config_synapse_drivers(h, hicann)
        self.config_synapse_switch(h, hicann)
        self.config_crossbar_switches(h, hicann)
        self.config_repeater(h, hicann)
        self.config_merger_tree(h, hicann)
        self.config_dncmerger(h, hicann)
        self.config_background_generators(h, hicann)
        self.flush_fpga(f)
        self.lock_repeater(h, hicann)

        self.config_synapse_array(h, hicann)

        self.config_neuron_config(h, hicann)
        self.config_neuron_quads(h, hicann)
        self.config_analog_readout(h, hicann)
        self.flush_fpga(f)


class StHALContainer(object):
    """Contains StHAL objects for hardware access. Multiple experiments can share one container."""
    def __init__(self, coord_wafer=pyhalbe.Coordinate.Wafer(),
                 coord_hicann=pyhalbe.Coordinate.HICANNOnWafer(pyhalbe.geometry.Enum(1)),
                 coord_analog=pyhalbe.Coordinate.AnalogOnHICANN(0)):

        wafer = pysthal.Wafer(coord_wafer)  # Stateful HICANN Container
        wafer.allocateHICANN(coord_hicann)
        hicann = wafer[coord_hicann]

        wafer.connect(pysthal.MagicHardwareDatabase())

        # analogRecorder() MUST be called after wafer.connect()
        adc = hicann.analogRecorder(coord_analog)
        adc.setReadoutTime(0.01)

        self.wafer = wafer
        self.hicann = hicann
        self.adc = adc
        self._cfg = WriteFGTwiceConfigurator()
        self._cfg_analog = UpdateAnalogOutputConfigurator()

    def write_config(self):
        """Write full configuration."""
        self.wafer.configure(self._cfg)

    def switch_analog_output(self, neuron_id, l1address=0, analog=0):
        """Write analog output configuration (only)."""
        coord_neuron = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(neuron_id))
        self.hicann.enable_l1_output(coord_neuron, pyhalbe.HICANN.L1Address(l1address))
        self.hicann.enable_aout(coord_neuron, pyhalbe.Coordinate.AnalogOnHICANN(analog))
        self.wafer.configure(self._cfg_analog)
