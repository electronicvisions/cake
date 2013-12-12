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
                 coord_hicann=pyhalbe.Coordinate.HICANNOnWafer(pyhalbe.Coordinate.Enum(280)),
                 coord_analog=pyhalbe.Coordinate.AnalogOnHICANN(0),
                 recording_time=1.e-4):
        """Initialize StHAL. kwargs default to vertical setup configuration."""

        wafer = pysthal.Wafer(coord_wafer)  # Stateful HICANN Container
        hicann = wafer[coord_hicann]

        self.wafer = wafer
        self.hicann = hicann
        self.adc = None
        self.recording_time = recording_time
        self.coord_analog = coord_analog
        self._connected = False
        self._cfg = WriteFGTwiceConfigurator()
        self._cfg_analog = UpdateAnalogOutputConfigurator()

    def connect(self):
        """Connect to the hardware."""
        self.wafer.connect(pysthal.MagicHardwareDatabase())
        self.connect_adc()
        self._connected = True

    def connect_adc(self):
        """Gets ADC handle"""
        # analogRecoorder() MUST be called after wafer.connect()
        if self.adc:
            self.adc.freeHandle()
            self.adc = None
        adc = self.hicann.analogRecorder(self.coord_analog)
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
        self.wafer.configure(self._cfg)

    def switch_analog_output(self, neuron_id, l1address=0, analog=0):
        """Write analog output configuration (only)."""
        if not self._connected:
            self.connect()
        coord_neuron = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(neuron_id))
        self.hicann.enable_l1_output(coord_neuron, pyhalbe.HICANN.L1Address(l1address))
        self.hicann.enable_aout(coord_neuron, pyhalbe.Coordinate.AnalogOnHICANN(analog))
        self.wafer.configure(self._cfg_analog)

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
