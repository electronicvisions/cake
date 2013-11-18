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
    def config_floating_gates(self, handle, hicann):
        """Write floating gates twice"""
        pyhalbe.HICANN.set_fg_values(handle, hicann.floating_gates)
        pyhalbe.HICANN.set_fg_values(handle, hicann.floating_gates)


class StHALContainer(object):
    """Contains StHAL objects for hardware access. Multiple experiments can share one container."""
    def __init__(self, coord_wafer=pyhalbe.Coordinate.Wafer(),
                 coord_hicann=pyhalbe.Coordinate.HICANNOnWafer(pyhalbe.geometry.Enum(280)),
                 coord_analog=pyhalbe.Coordinate.AnalogOnHICANN(0),
                 recording_time=1.e-4):
        """Initialize StHAL, connect to hardware. kwargs default to vertical setup configuration."""

        wafer = pysthal.Wafer(coord_wafer)  # Stateful HICANN Container
        hicann = wafer[coord_hicann]

        wafer.connect(pysthal.MagicHardwareDatabase())

        # analogRecorder() MUST be called after wafer.connect()
        adc = hicann.analogRecorder(coord_analog)
        adc.setRecordingTime(recording_time)

        self.wafer = wafer
        self.hicann = hicann
        self.adc = adc
        self._cfg = WriteFGTwiceConfigurator()
        self._cfg_analog = UpdateAnalogOutputConfigurator()

    def disconnect(self):
        """Free handles."""
        self.wafer.disconnect()
        self.adc.freeHandle()

    def write_config(self):
        """Write full configuration."""
        self.wafer.configure(self._cfg)

    def switch_analog_output(self, neuron_id, l1address=0, analog=0):
        """Write analog output configuration (only)."""
        coord_neuron = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(neuron_id))
        self.hicann.enable_l1_output(coord_neuron, pyhalbe.HICANN.L1Address(l1address))
        self.hicann.enable_aout(coord_neuron, pyhalbe.Coordinate.AnalogOnHICANN(analog))
        self.wafer.configure(self._cfg_analog)
