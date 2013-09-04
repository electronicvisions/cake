import pysthal


class UpdateAnalogOutputConfigurator(pysthal.HICANNConfigurator):
    def config(self, fpga_handle, h, hicann):
        self.config_neuron_quads(h, hicann)
        self.config_analog_readout(h, hicann)
        self.flush_fpga(fpga_handle)