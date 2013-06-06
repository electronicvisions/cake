"""Cairo interface to HALbe via pyhalbe. Adresses a single HICANN."""

import pyhalbe
from pyhalbe.geometry import Enum

import numpy as np

import pycairo.interfaces.adc
import pycairo.logic.helpers
import pycairo.config.hardware as config
from pycairo.config.default_hardware_params import get_global_parameters, get_HW_parameters


class HWNeurons(object):
    def __init__(self, default=pyhalbe.HICANN.NeuronQuad()):
        self.quads = [pyhalbe.HICANN.NeuronQuad(default) for ii in range(512 / pyhalbe.HICANN.NeuronQuad.size())]

    def enable_aout(self, neuron):
        n = self._get_neuron(neuron)
        n.enable_aout(True)

    def disable_aout(self):
        for q in self.quads:
            for ii in range(pyhalbe.HICANN.NeuronQuad.size()):
                q[pyhalbe.Coordinate.NeuronOnQuad(Enum(ii))].enable_aout(False)

    def write(self, handle):
        for ii, q in enumerate(self.quads):
            quad = pyhalbe.Coordinate.QuadOnHICANN(ii)
            pyhalbe.HICANN.set_denmem_quad(handle, quad, q)

    def _get_neuron(self, neuron_id):
        n = pyhalbe.Coordinate.NeuronOnHICANN(Enum(neuron_id))
        quad = self.quads[int(n.quad())]
        quad_neuron = n.neuronOnQuad()
        return quad[quad_neuron]


class HalbeInterface:
    def __init__(self, hicann_id, setup_ip, setup_port=config.FPGA_PORT):
        """Connect to a vertical setup at a given IP.

        Vars:
            hicann_id: HICANN id
            setup_ip (str): string containing of IPv4 adress
            setup_port (int): port
        """

        self.helpers = pycairo.logic.helpers.Helpers()
        self.adc = pycairo.interfaces.adc.ADCInterface()  # TODO WSS case?

        if type(setup_ip) is tuple:  # old format, tuple of IP and port
            self.ip = pyhalbe.Coordinate.IPv4.from_string(setup_ip[0])
            self.port = int(setup_ip[1])
        else:
            self.ip = pyhalbe.Coordinate.IPv4.from_string(setup_ip)
            self.port = int(setup_port)
        print 'Connecting to ' + self.ip.to_string() + ':' + str(self.port)

        highspeed = True
        arq = True
        self.myPowerBackend = pyhalbe.PowerBackend.instanceVerticalSetup()
        self.myPowerBackend.SetupReticle(highspeed, self.ip, self.port, 1, arq)
        self.h = pyhalbe.Handle.HICANN(pyhalbe.Coordinate.HICANNGlobal(Enum(hicann_id)))
        self.dnc = pyhalbe.Coordinate.DNCGlobal(Enum(0))

        self.init_HW()

    def convert_to_voltage_fg(self, value):
        fgvalue = int(float(value) / config.max_v * config.res_fg)
        if fgvalue < 0 or fgvalue > config.res_fg:
            raise ValueError("Floating gate value {} out of range".format(fgvalue))
        return fgvalue

    def convert_to_current_fg(self, value):
        fgvalue = int(float(value) / config.max_i * config.res_fg)
        if fgvalue < 0 or fgvalue > config.res_fg:
            raise ValueError("Floating gate value {} out of range".format(fgvalue))
        return fgvalue

    def send_fg_configure(self, neurons, parameters):
        """Main configuration function. Erase and write FGarray and read out neuron neuronID.

        Args:
            neuron_index The index of neurons. For example : [[1, 2, 3], [1, 2, 3]]
            parameter The parameter to measure. Example : "EL"
        """

        if len(set(p["Vreset"] for p in parameters)) != 1:
            raise RuntimeError("Vreset must be equal for all neurons")

        g_p = get_global_parameters()
        g_p["V_reset"] = self.convert_to_voltage_fg(parameters[0]['Vreset'])
        params = [dict((k, 0) for k in get_HW_parameters()) for ii in range(pyhalbe.FGControl.number_neurons)]

        for ii, neuron in enumerate(neurons):
            params[neuron] = parameters[ii]

        self.write_fg(g_p, params)

    def write_fg(self, global_parameters, parameters):
        assert len(parameters) == pyhalbe.FGControl.number_neurons
        fgc = pyhalbe.FGControl()

        for ii in range(pyhalbe.FGControl.number_blocks):  # number_blocks == 4
            fg_block = pyhalbe.Coordinate.FGBlockOnHICANN(ii)
            fgc.setConfig(fg_block, pyhalbe.HICANN.FGConfig())
            g_p = dict(global_parameters)  # copy dictionary before deleting elements
            if ii in (0, 2):
                # Remove left-only global parameters
                del g_p["V_clrc"]
                del g_p["V_bexp"]
            else:
                # Remove right-only global parameters
                del g_p["V_clra"]
                del g_p["V_bout"]
            for name, value in g_p.iteritems():
                fgc.setShared(fg_block, getattr(pyhalbe.HICANN, name), value)

        for ii in range(pyhalbe.FGControl.number_neurons):
            p = parameters[ii]
            neuron = pyhalbe.Coordinate.NeuronOnHICANN(Enum(ii))
            fgc.setNeuron(neuron, pyhalbe.HICANN.E_l,        self.convert_to_voltage_fg(p['EL']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.E_syni,     self.convert_to_voltage_fg(p['Esyni']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.E_synx,     self.convert_to_voltage_fg(p['Esynx']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.I_bexp,     self.convert_to_current_fg(p['Ibexp']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.I_convi,    self.convert_to_current_fg(p['gsyni']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.I_convx,    self.convert_to_current_fg(p['gsynx']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.I_fire,     self.convert_to_current_fg(p['b']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.I_gl,       self.convert_to_current_fg(p['gL']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.I_gladapt,  self.convert_to_current_fg(p['a']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.I_intbbi,   self.convert_to_current_fg(p['Iintbbi']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.I_intbbx,   self.convert_to_current_fg(p['Iintbbx']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.I_pl,       self.convert_to_current_fg(p['tauref']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.I_radapt,   self.convert_to_current_fg(p['tw']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.I_rexp,     self.convert_to_current_fg(p['dT']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.I_spikeamp, self.convert_to_current_fg(p['Ispikeamp']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.V_exp,      self.convert_to_voltage_fg(p['Vexp']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.V_syni,     self.convert_to_voltage_fg(p['Vsyni']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.V_synx,     self.convert_to_voltage_fg(p['Vsynx']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.V_syntci,   self.convert_to_voltage_fg(p['tausyni']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.V_syntcx,   self.convert_to_voltage_fg(p['tausynx']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.V_t,        self.convert_to_voltage_fg(p['Vt']))

        for fg_block in [pyhalbe.Coordinate.FGBlockOnHICANN(ii) for ii in range(pyhalbe.FGControl.number_blocks)]:
            # setting PLL frequency is disabled
            # because it is not well understood
            #pyhalbe.HICANN.set_PLL_frequency(self.h, config.fg_pll)
            pyhalbe.HICANN.set_fg_values(self.h, fgc.extractBlock(fg_block))  # write 3(!) times for better accuracy
#            pyhalbe.HICANN.set_fg_values(self.h, fgc.extractBlock(fg_block)
#            pyhalbe.HICANN.set_fg_values(self.h, fgc.extractBlock(fg_block)
            # setting PLL frequency is disabled
            # because it is not well understood
            #pyhalbe.HICANN.set_PLL_frequency(self.h, config.pll)

    ## Erase FGArra
    def erase_fg(self):
        g_p = dict((k, 0) for k in get_global_parameters())
        p = [dict((k, 0) for k in get_HW_parameters()) for ii in range(pyhalbe.FGControl.number_neurons)]
        self.write_fg(g_p, p)

    def sweep_neuron(self, neuron, side=None, current=0):
        """Sweep output to a given neuron number
            i: The neuron number
            side: The side of the chip, can be "top" or "bottom"
            current: The current to inject in the neuron, values from 0 to 1023
        """

        neuron = int(neuron)
        assert neuron >= 0 and neuron < 512
        assert current == 0

        analog_channel = pyhalbe.Coordinate.AnalogOnHICANN(0)
        odd = bool(neuron % 2)
        ac = pyhalbe.HICANN.Analog()
        if neuron < 256:
            if odd:
                ac.set_membrane_top_odd(analog_channel)
            else:
                ac.set_membrane_top_even(analog_channel)
        else:
            if odd:
                ac.set_membrane_bot_odd(analog_channel)
            else:
                ac.set_membrane_bot_even(analog_channel)
        pyhalbe.HICANN.set_analog(self.h, ac)

        neurons = HWNeurons()
        neurons.disable_aout()
        neurons.write(self.h)
        neurons.enable_aout(neuron)
        neurons.write(self.h)

    def set_one_neuron_current(self, neuron, current):
        """Configure one neuron only

        Args:
            neuron: The neuron to configure
            current: The current to inject, values from 0 to 1023
        """

        keys = ["q", str(current), "n", str(neuron)]
        self.call_testmode('tm_neuron', keys, ['-c', self.xmlfile])

    def set_stimulus(self, i):
        """Set stimulus

        Args:
            i: Set the current stimulus to the value i
        """
        # Convert to digital value
        self.configure_hardware(["s", str(i), "x"])

    def set_ramp_stimulus(self, i):
        """Set ramp stimulus

        Args:
            i: Set the current stimulus to the value i
        """
        # Convert to digital value
        self.configure_hardware(["q", str(i), "x"])

    def set_constant_stimulus(self, i):
        """Set stimulus

        Args:
            i Set the current stimulus to the value i
        """
        # Convert to digital value
        self.configure_hardware(["s", str(i), "x"])

    def set_stimulus_nA(self, i):
        """Set stimulus in nA

        Args:
            i: Set the current stimulus to the value i, in nA
        """

        # Calculate FG value
        # FIXME hardcoded values
        a = -0.0007
        b = 0.56
        c = 2.94
        i = a * i * i + b * i + c

        # Convert to digital value
        self.configure_hardware(["q", str(i), "x"])

    def init_HICANN(self):
        print "hicann reset.. ",
        pyhalbe.HICANN.full_reset(self.h, False)
        print "done"

        # setting PLL frequency is disabled
        # because it is not well understood
        #print "Setting PLL to", config.pll
        #pyhalbe.HICANN.set_PLL_frequency(self.h, config.pll)

    def init_HW(self):
        """Init iBoard and HICANN"""
        print "Full hicann reset...",
        pyhalbe.HICANN.full_reset(self.h, True)
        print "done"

        self.init_HICANN()

    def activate_BEG(self, BEG_status, BEG_type, cycles, neuron):
        """Activate background event generator.

        Args:
            BEG_status: The status of the BEG, can be "ON" or "OFF"
            BEG_type: The mode of the BEG, can be "REGULAR" or "POISSON"
            cycles: The number of cycles of the BEG
            neuron: The neuron to stimulate witht the BEG
        """

        # Key file
        if (BEG_status == 'ON'):
            if (BEG_type == 'REGULAR'):
                keys = ['f', 'h', str(neuron), str(cycles), 'H', str(cycles), '0']
            if (BEG_type == 'POISSON'):
                keys = ['f', 'h', str(neuron), str(cycles), 'H', str(cycles), '1', '100']
        if (BEG_status == 'OFF'):
            keys = ['7']

        self.configure_hardware(keys + ["x"])

    def deactivate_BEG(self):
        """Deactivate background event generator"""
        self.configure_hardware(["0", "x"])

    def calc_ISI(self, spikes_list):
        """Calc ISI

        Args:
            spikes_list: The input spikes list
        """

        ISI = []
        for i in np.arange(1, len(spikes_list)):
            ISI.append(spikes_list[i]-spikes_list[i-1])

        return ISI

    def calc_freq(self, spikes_list):
        """Calc frequency

        Args:
            spikes_list: The input spikes list
        """

        ISI = self.calc_ISI(spikes_list)
        mean_ISI = np.mean(ISI)
        if (mean_ISI != 0):
            return 1/mean_ISI
        else:
            return 0

    def calc_std(self, spikes_list):
        """Calculate standard deviation of the spiking frequencies

        Args:
            spikes_list: The input spikes list
        """

        ISI = self.calc_ISI(spikes_list)
        freqs = []
        for i in ISI:
            if (i != 0):
                freqs.append(1/i)
        std_freq = np.std(freqs)
        return std_freq

    def switch_neuron(self, current_neuron):
        """Switch to the correct hardware neuron.

        Args:
            current_neuron: number of the desired neuron."""

        if(current_neuron < 128):
            self.sweep_neuron(current_neuron, 'top')
            side = 'left'
        elif(current_neuron > 127 and current_neuron < 256):
            self.sweep_neuron(-current_neuron + 383, 'top')
            side = 'right'
        elif(current_neuron > 255 and current_neuron < 384):
            self.sweep_neuron(current_neuron, 'bottom')
            side = 'left'
        elif(current_neuron > 383):
            self.sweep_neuron(-current_neuron + 895, 'bottom')
            side = 'right'

        return side

    def measure(self, neurons, parameter, parameters, stimulus=0, spike_stimulus=[]):
        """Main measurement function.

        Args:
            neuron_index The index of neurons. For example : [[1, 2, 3], [1, 2, 3]]
            parameter The parameter to measure. Example : "EL"
            parameters The neuron parameters
            stimulus Value of the current stimulus
            value The hardware value of the parameter that is currently being measured
            spike_stimulus The list of spikes to send to the hardware
        """

        meas_array = []  # measurement array
        for n, current_neuron in enumerate(neurons):
            print "Measuring neuron " + str(current_neuron)
            self.switch_neuron(current_neuron)

            # Parameter specific measurement
            if (parameter == 'EL'):
                measured_value = self.adc.get_mean() * 1000  # ADC interface returns volts, multiply by 1000 for millivolts
                print "Measured value for EL : " + str(measured_value) + " mV"
                meas_array.append(measured_value)
            elif (parameter == 'Vreset'):
                measured_value = self.adc.get_min() * 1000
                print "Measured value for Vreset : " + str(measured_value) + " mV"
                meas_array.append(measured_value)
            elif (parameter == 'Vt'):
                measured_value = self.adc.get_max() * 1000
                print "Measured value for Vt : " + str(measured_value) + " mV"
                meas_array.append(measured_value)
            elif (parameter == 'gL' or parameter == 'tauref' or parameter == 'a' or parameter == 'b' or parameter == 'Vexp'):
                measured_value = self.adc.get_freq()
                print "Measured frequency : " + str(measured_value) + " Hz"
                meas_array.append(measured_value)
            elif (parameter == 'tw'):
                # Inject current
                self.set_stimulus(config.current_default)

                # Get trace
                trace = self.adc.read_adc(config.sample_time_tw)
                t, v = trace.time, trace.voltage

                # Apply fit
                tw = self.helpers.tw_fit(t, v, parameters['C'], parameters['gL'])

                print "Measured value : " + str(tw)
                meas_array.append(tw)
            elif (parameter == 'dT'):
                # Get trace
                trace = self.adc.read_adc(config.sample_time_dT)
                t, v = trace.time, trace.voltage

                # Calc dT
                dT = self.helpers.calc_dT(t, v, parameters['C'], parameters['gL'], parameters['EL'])

                print "Measured value : " + str(dT)
                meas_array.append(dT)
            elif parameter in ('tausynx', 'tausyni'):
                # Activate BEG
                self.activate_BEG('ON', 'REGULAR', 2000, current_neuron)
                print 'BEG active'

                # Get trace
                t, v = self.adc.adc_sta(20e-6)

                # Convert v with scaling
                for i in range(len(v)):
                    t[i] = t[i]*1000

                if (parameter == 'tausynx'):
                    fit = self.helpers.fit_PSP(t, v, parameter, parameters[parameter], parameters['Esynx'])
                if (parameter == 'tausyni'):
                    fit = self.helpers.fit_PSP(t, v, parameter, parameters[parameter], parameters['Esyni'])

                print "Measured value : " + str(fit)
                meas_array.append(fit)

        return meas_array
