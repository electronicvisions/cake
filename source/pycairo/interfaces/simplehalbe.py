# -*- coding: utf-8 -*-

"""Simple wrapper for HALbe.
Initializes a single HICANN, allows writing floating gates at runtime and performing ADC measurements on single neurons."""

import pyhalbe
import pycalibtic
import numpy as np


class ActiveConnection(object):
    """Initializes the hardware for multiple measurements.
    Keeps handles and initial HICANN configuration.
    Floating gate values can be modified at runtime."""

    def __init__(self, ip, port, adc_board_id, adc_input_channel, adc_trigger_channel, vertical_setup=True, hicann_id=0, dnc_id=0, fpga_id=0):
        """Grab required handles and initialize HICANN."""

        coord_ip = pyhalbe.Coordinate.IPv4.from_string(ip)
        port = int(port)
        # TODO get adc_input_channel, adc_trigger_channel from cable db,
        # maybe even get ip and port

        if vertical_setup:
            pb = pyhalbe.PowerBackend.instanceVerticalSetup()
        else:
            pb = pyhalbe.PowerBackend.instanceWafer()

        highspeed = True
        arq = True
        pb.SetupReticle(highspeed, coord_ip, port, 1, arq)

        handle_hicann = pyhalbe.Handle.HICANN(pyhalbe.Coordinate.HICANNGlobal(pyhalbe.geometry.Enum(hicann_id)))

        # TODO coord_dnc is not used?
        #coord_dnc = pyhalbe.Coordinate.DNCGlobal(pyhalbe.geometry.Enum(dnc_id))

        self.handle_fpga = pyhalbe.Handle.FPGA(pyhalbe.Coordinate.FPGAGlobal(pyhalbe.geometry.Enum(fpga_id)))

        reset_synapses = True  # also reset synapses, needs more time
        pyhalbe.HICANN.full_reset(handle_hicann, reset_synapses)  # does not reset floating gate values
        pyhalbe.HICANN.init(handle_hicann)  # initialize communication route, run after each reset!
        self.handle_hicann = handle_hicann

        self.handle_adc = pyhalbe.Handle.ADC()
        self.adc_input_channel = adc_input_channel
        self.adc_trigger_channel = adc_trigger_channel

        # load ADC calibration data for the board in use
        backend_adc = pycalibtic.loadBackend(pycalibtic.loadLibrary("libcalibtic_mongo.so"))
        backend_adc.config("host", "cetares")
        backend_adc.config("collection", "adc")
        backend_adc.init()
        coord_adc = pyhalbe.Coordinate.ADC(adc_board_id)
        calib_adc = pycalibtic.ADCCalibration()
        calib_adc.load(backend_adc, coord_adc)
        self.calib_adc = calib_adc

    def activate_neuron(self, neuron_id=0):
        """Activate firing of a single neuron and connect it to analog output."""
        # TODO disable other neurons (at least aout)
        nconf = pyhalbe.HICANN.NeuronConfig()
        for side in (int(pyhalbe.geometry.TOP), int(pyhalbe.geometry.BOTTOM)):
            # use big capacitance
            nconf.bigcap[side] = True
            # use fastest membrane possible
            nconf.fast_I_gl[side] = True
            nconf.slow_I_gl[side] = False
        pyhalbe.HICANN.set_neuron_config(self.handle_hicann, nconf)

        nquad = pyhalbe.HICANN.NeuronQuad()
        nactive = pyhalbe.HICANN.Neuron()

        nactive.activate_firing(True)
        nactive.enable_fire_input(False)
        nactive.enable_current_input(False)
        nactive.enable_aout(True)
        # there are limited combinations of these options.
        # see HICANN doc 2012-09-13, p. 32, last table in 4.4.4

        if (neuron_id > 255):
            # this configures only a single neuron
            nquad[pyhalbe.Coordinate.NeuronOnQuad(pyhalbe.Coordinate.X(neuron_id % 2), pyhalbe.Coordinate.Y(1))] = nactive
            # to configure all neurons in this quad, assign Neuron instances to the other quad neurons as well
            pyhalbe.HICANN.set_denmem_quad(self.handle_hicann, pyhalbe.Coordinate.QuadOnHICANN((neuron_id-256)/2), nquad)
        else:
            nquad[pyhalbe.Coordinate.NeuronOnQuad(pyhalbe.Coordinate.X(neuron_id % 2), pyhalbe.Coordinate.Y(0))] = nactive
            pyhalbe.HICANN.set_denmem_quad(self.handle_hicann, pyhalbe.Coordinate.QuadOnHICANN(neuron_id/2), nquad)

        # flush configuration
        pyhalbe.FPGA.start(self.handle_fpga)

    def enable_analog_output(self, neuron_id, aout_id=0):
        # TODO disable previous aout?
        aout = pyhalbe.HICANN.Analog()
        coord_aout = pyhalbe.Coordinate.AnalogOnHICANN(aout_id)
        aout.enable(coord_aout)
        if (neuron_id > 255):
            if (neuron_id % 2):
                aout.set_membrane_bot_odd(coord_aout)
            else:
                aout.set_membrane_bot_even(coord_aout)
        else:
            if (neuron_id % 2):
                aout.set_membrane_top_odd(coord_aout)
            else:
                aout.set_membrane_top_even(coord_aout)
        pyhalbe.HICANN.set_analog(self.handle_hicann, aout)

    def write_floating_gates(self, neuron_id, neuron_parameters, shared_parameters):
        """Write floating gate parameters of a single neuron and parameters shared between neurons."""

        fgc = pyhalbe.FGControl()

        for neuron_id in range(512):
            coord = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.geometry.Enum(neuron_id))
            for parameter in neuron_parameters:
                param = getattr(pyhalbe.HICANN.neuron_parameter, parameter)
                value = neuron_parameters[parameter]
                fgc.setNeuron(coord, param, value)

        for fgblock in range(4):
            coord = pyhalbe.Coordinate.FGBlockOnHICANN(fgblock)
            for parameter in shared_parameters:
                if parameter in ('V_clrc', 'V_bexp') and fgblock in (0, 2):
                    # right only parameter in left block, skip
                    continue
                if parameter in ('V_clra', 'V_bout') and fgblock in (1, 3):
                    # left only parameter in right block, skip
                    continue
                param = getattr(pyhalbe.HICANN.shared_parameter, parameter)
                value = shared_parameters[parameter]
                fgc.setShared(coord, param, value)

        # program multiple times for better accuracy
        for repetition in range(2):
            for fgblock in range(4):
                pyhalbe.HICANN.set_fg_values(self.handle_hicann, fgc.extractBlock(pyhalbe.Coordinate.FGBlockOnHICANN(fgblock)))

        # flush configuration
        pyhalbe.FPGA.start(self.handle_fpga)

    def measure_adc(self, trace_length):
        """Trigger ADC recording. Converts raw data to voltage using ADC calibration data. Adds time.

        Returns:
            numpy arrays of time [s], voltage [V]
        """

        handle_adc = self.handle_adc
        conf_adc = pyhalbe.ADC.Config(trace_length, self.adc_input_channel, self.adc_trigger_channel)
        pyhalbe.ADC.config(handle_adc, conf_adc)
        pyhalbe.ADC.trigger_now(handle_adc)
        raw_trace = pyhalbe.ADC.get_trace(handle_adc)
        voltage = self.calib_adc.apply(int(self.adc_input_channel), np.array(raw_trace, dtype=np.ushort))
        # calculate time in seconds
        # TODO replace magic conversion factor by database entry
        time = np.arange(len(voltage)) * 0.00000001733853
        return time, voltage
