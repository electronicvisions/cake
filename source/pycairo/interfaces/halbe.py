'''Cairo interface to HALbe via pyhalbe'''

import os
import numpy
import pylab
import sys

import pyhalbe
from pycairo.config.default_hardware_params import get_global_parameters, get_HW_parameters

class HWNeurons(object):
    def __init__(self, default = pyhalbe.HICANN.NeuronQuad()):
        self.quads = [pyhalbe.HICANN.NeuronQuad(default) for ii in range(512/pyhalbe.HICANN.NeuronQuad.size())]

    def enable_aout(self, neuron):
        n = self._get_neuron(neuron)
        n.enable_aout(True)

    def disable_aout(self):
        for q in self.quads:
            for ii in range(pyhalbe.HICANN.NeuronQuad.size()):
                q[pyhalbe.Coordinate.NeuronOnQuad(ii)].enable_aout(False)

    def write(self, handle):
        for ii, q in enumerate(self.quads):
            quad = pyhalbe.Coordinate.QuadOnHICANN(ii)
            pyhalbe.HICANN.set_denmem_quad(handle, quad, q)

    def _get_neuron(self, neuron_id):
        n = pyhalbe.Coordinate.NeuronOnHICANN(neuron_id)
        quad = self.quads[int(n.quad())] # TODO replace int() by n.quad().id() when id() is available
        quad_neuron = n.neuronOnQuad()
        return quad[quad_neuron]


class HalbeInterface:
    def __init__(self, hicann, ip):
        '''Connect to a vertical setup at a given IP
        
        Vars:
            hicann: HICANN id
            ip: tuple (str, str or int) of IPv4 adress and port
        '''
        Enum = pyhalbe.geometry.Enum

        # Floating gate parameters
        self.res_fg = 1023
        self.max_v = 1800 # mV
        self.max_i = 2500 # nA

        self.pll = 150
        self.fg_pll = 100

        print 'Connecting to ' + str(ip)
        highspeed = True
        arq = True
        self.ip = pyhalbe.Coordinate.IPv4.from_string(ip[0])
        self.port = int(ip[1])
        self.myPowerBackend = pyhalbe.PowerBackend.instanceVerticalSetup()
        self.myPowerBackend.SetupReticle(highspeed, self.ip, self.port, 1, arq)
        self.h = pyhalbe.Handle.HICANN(pyhalbe.Coordinate.HICANNGlobal(Enum(hicann)))
        self.dnc = pyhalbe.Coordinate.DNCGlobal(Enum(0))

        self.init_HW()


    def convert_to_voltage_fg(self,value):
        fgvalue = int(float(value)/self.max_v*self.res_fg)
        if fgvalue < 0 or fgvalue > self.res_fg:
            raise ValueError("Floating gate value {} out of range".format(fgvalue))
        return fgvalue

    def convert_to_current_fg(self,value):
        fgvalue = int(float(value)/self.max_i*self.res_fg)
        if fgvalue < 0 or fgvalue > self.res_fg:
            raise ValueError("Floating gate value {} out of range".format(fgvalue))
        return fgvalue

    ## Create XML file, erase and write FGarray and read out neuron neuronID
    # @param hicann_index The index of HICANNs
    # @param neuron_index The index of neurons
    # @param parameters The neuron parameters
    # @param option The configuration option. Can be "XML_only" or "configure"
    def send_fg_configure(self, neurons, parameters, option='configure'):
        if len(set(p["Vreset"] for p in parameters)) != 1:
            raise RuntimeError("Vreset must be equal for all neurons")

        g_p = get_global_parameters()
        g_p["V_reset"] = self.convert_to_voltage_fg(parameters[0]['Vreset'])
        params = [ dict( (k, 0) for k in get_HW_parameters() ) for ii in range(pyhalbe.FGControl.number_neurons)]

        for ii, neuron in enumerate(neurons):
            params[neuron] = parameters[ii]

        self.write_fg(g_p, params)

    def write_fg(self, global_parameters, parameters):
        assert len(parameters) == pyhalbe.FGControl.number_neurons
        fgc = pyhalbe.FGControl()

        for fg_block in [pyhalbe.Coordinate.FGBlockOnHICANN(ii) for ii in range(pyhalbe.FGControl.number_blocks)]:
            fgc.setConfig(fg_block, pyhalbe.HICANN.FGConfig())
            g_p = dict(global_parameters)
            if fg_block == pyhalbe.Coordinate.FGBlockOnHICANN(0) or fg_block == pyhalbe.Coordinate.FGBlockOnHICANN(2):
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
            neuron = pyhalbe.Coordinate.NeuronOnHICANN(ii)
            fgc.setNeuron(neuron, pyhalbe.HICANN.E_l,        self.convert_to_voltage_fg(p['EL']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.E_syni,     self.convert_to_voltage_fg(p['Esyni']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.E_synx,     self.convert_to_voltage_fg(p['Esynx']))
            fgc.setNeuron(neuron, pyhalbe.HICANN.I_bexp,     self.convert_to_current_fg(p['expAct']))
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
            pyhalbe.HICANN.set_PLL_frequency(self.h, self.fg_pll)
            pyhalbe.HICANN.set_fg_values(self.h, fgc.extractBlock(fg_block)) # write 3(!) times for better accuracy
#            pyhalbe.HICANN.set_fg_values(self.h, fgc.extractBlock(fg_block)
#            pyhalbe.HICANN.set_fg_values(self.h, fgc.extractBlock(fg_block)
            pyhalbe.HICANN.set_PLL_frequency(self.h, self.pll)

    ## Erase FGArra
    def erase_fg(self):
        g_p = dict( (k, 0) for k in get_global_parameters() )
        p = [ dict( (k, 0) for k in get_HW_parameters() ) for ii in range(pyhalbe.FGControl.number_neurons)]
        self.write_fg(g_p, p)

    ## Sweep output to a given neuron number
    # @param i The neuron number
    # @param side The side of the chip, can be "top" or "bottom"
    # @param current The current to inject in the neuron, values from 0 to 1023    
    def sweep_neuron(self, neuron, side = None, current=0):
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



    ## Configure one neuron only
    # @param neuron The neuron to configure
    # @param current The current to inject, values from 0 to 1023
    def set_one_neuron_current(self,neuron,current):
        keys = ["q", str(current), "n", str(neuron)]
        self.call_testmode('tm_neuron', keys, ['-c', self.xmlfile])

    ## Set stimulus
    # @param i Set the current stimulus to the value i
    def set_stimulus(self,i):
        # Convert to digital value
        self.configure_hardware(["s", str(i), "x"])

    ## Set ramp stimulus
    # @param i Set the current stimulus to the value i
    def set_ramp_stimulus(self,i):
        # Convert to digital value
        self.configure_hardware(["q", str(i), "x"])

    ## Set stimulus
    # @param i Set the current stimulus to the value i
    def set_constant_stimulus(self,i):
        # Convert to digital value
        self.configure_hardware(["s", str(i), "x"])

    ## Set stimulus in nA
    # @param i Set the current stimulus to the value i, in nA
    def set_stimulus_nA(self,i):

        # Calculate FG value
        a = -0.0007
        b = 0.56
        c = 2.94
        i = a*i*i+b*i+c

        # Convert to digital value
        self.configure_hardware(["q", str(i), "x"])

    ## Init HICANN, todo rename?
    def init_L1(self):
        print "hicann reset.. ",
        pyhalbe.HICANN.full_reset(self.h, False);
        print "done"

        print "Setting PLL to", self.pll
        pyhalbe.HICANN.set_PLL_frequency(self.h, self.pll)

    ## Init iBoard and HICANN
    def init_HW(self):
        print "Full hicann reset...",
        pyhalbe.HICANN.full_reset(self.h, True);
        print "done"

        self.init_L1()

        #print "Clearing floating gates array... ",
        #self.erase_fg()
        #print "done"


#        # Reset JTAG and HICANN
#        keys = ['1', '2', '3']
#        # Set voltages on iBoard and analog out
#        keys += ['7', 'c', 'x']
#
#        self.call_testmode('tmak_iboardv2', keys)
#        print "iBoard configured"  
#
#        # Clear synapse array
#        self.configure_hardware(["0","x"])
#        print "Synapse array cleared"


    ## Activate background event generator
    # @param BEG_status The status of the BEG, can be "ON" or "OFF"
    # @param BEG_type The mode of the BEG, can be "REGULAR" or "POISSON"
    # @param cycles The number of cycles of the BEG
    # @param neuron The neuron to stimulate witht the BEG
    def activate_BEG(self,BEG_status,BEG_type,cycles,neuron):

        # Key file
        if (BEG_status == 'ON'):
            if (BEG_type == 'REGULAR'):
                keys = ['f','h',str(neuron),str(cycles),'H',str(cycles),'0']
            if (BEG_type == 'POISSON'):
                keys = ['f','h',str(neuron),str(cycles),'H',str(cycles),'1','100']
        if (BEG_status == 'OFF'):
            keys = ['7']

        # Launch test mode
        self.configure_hardware(keys + ["x"])

    ## Deactivate background event generator
    def deactivate_BEG(self):
        self.configure_hardware(["0", "x"])

    ## Read all spikes
    # @param neuron_index The neuron index
    # @param range The number of neuron circuits to read from at a time
    def read_spikes_freq(self,neuron_index,range=4):

        def measure_top_half(neuron_index,range):
            neuron_list, freqs = self.read_spikes_half(neuron_index,range)

            freqs = list(freqs)
            for i,item in enumerate(neuron_index):
                if (item in neuron_list):
                    pass
                else:
                    freqs.insert(i,0)

            freqs_new = []

            print neuron_index
            print len(freqs), freqs
            print len(neuron_list), neuron_list

            for i,item in enumerate(neuron_index):

                if (item < 128):
                    freqs_new.append(freqs[i])

                if (item > 127 and item < 256):
                    freqs_new.append(freqs[-i+255+128])

            return neuron_index, freqs_new

    ## Read spikes from half the chip
    # @param neuron_index The neuron index
    # @param range The number of neuron circuits to read from at a time
    def read_spikes_half(self,neuron_index,range=4):

        keys = []
        for n in numpy.arange(min(neuron_index),max(neuron_index),range):
            keys.append('R')
            keys.append(str(n))
            keys.append(str(n+range-1))
        keys.append("x")

        # Remove spike file
        os.system("rm " + self.tm_path + 'train')   

        # Launch test mode
        self.configure_hardware(keys)

        # Read file
        try:
            data = numpy.genfromtxt(self.tm_path + "train")
        except:
            data = []

        # Neuron index
        neuron_list = []
        spikes_lists = []

        if (data != []):

            # Sort data
            for i in data:
                neuron_number = (-i[1]+7)*64 + i[2]

                if (not (neuron_number in neuron_list)):
                    neuron_list.append(int(neuron_number))

            # Organize spikes lists
            for i in neuron_list:
                spikes_lists.append([])

            for i in data:
                neuron_number = (-i[1]+7)*64 + i[2]
                spikes_lists[neuron_list.index(neuron_number)].append(i[0])

            # Calc freq for spikes_lists
            freqs = []
            err_freqs = []
            for i,item in enumerate(spikes_lists):
                frequency = self.calc_freq(item)
                if (frequency > 0):
                    freqs.append(frequency)
                else:
                    freqs.append(0)

            # Sort
            neuronsfreq = zip(neuron_list, freqs)
            neuronsfreq.sort()
            neuron_list, freqs = zip(*neuronsfreq)

            # Cut to match neuron_index
            measured_freq = []
            for i,item in enumerate(neuron_list):
                if item in neuron_index:
                    measured_freq.append(freqs[i])

            # Calculate new neuron list
            neuron_list_new = []
            for i in neuron_list:
                if (i > -1 and i < 32):
                    neuron_list_new.append(i)
                if (i > 95 and i < 160):
                    neuron_list_new.append(i-96+32)
                if (i > 223 and i < 288):
                    neuron_list_new.append(i-224+32+64)
                if (i > 351 and i < 416):
                    neuron_list_new.append(i-352+32+128)
                if (i > 479 and i < 512):
                    neuron_list_new.append(i-480+32+128+64)

        else:
            neuron_list_new = []
            freqs = []

        return neuron_list_new, freqs

    # Read spikes and plot for one half
    def plot_spikes(self,neuron_index,range=4):

        # Measure top half
        if (max(neuron_index) < 256):
            neuron_list, spikes = self.plot_spikes_half(neuron_index,range)

        # Measure bottom half
        if (min(neuron_index) > 255):
            temp_neuron_list, spikes = self.plot_spikes_half(neuron_index,range)
            # Add 256 to neuron list
            neuron_list = []
            for i in temp_neuron_list:
                neuron_list.append(i+256)

        # Measure all chip    
        if (max(neuron_index) > 256 and min(neuron_index) < 255):
            # Split index
            neuron_index_top = []
            neuron_index_bottom = []
            for i in neuron_index:
                if (i < 256):
                    neuron_index_top.append(i)
                else:
                    neuron_index_bottom.append(i)
            # Measure
            neuron_list_top, spikes_top = self.plot_spikes_half(neuron_index_top,range)
            neuron_list_bottom, spikes_bottom = self.plot_spikes_half(neuron_index_top,range)

            # Concatenate results
            new_neuron_list_bottom = []
            for i in neuron_list_bottom:
                new_neuron_list_bottom.append(i+256)

            neuron_list = neuron_list_top + new_neuron_list_bottom
            spikes = spikes_top + spikes_bottom

        for i,item in enumerate(neuron_list):
            pylab.scatter(spikes[i],neuron_list[i]*numpy.ones(len(spikes[i])),s=3)

    ## Read spikes and plot for one half
    # @param neuron_index The neuron index
    # @param range The number of neuron circuits to read from at a time
    def plot_spikes_half(self,neuron_index,range=4):

        keys = []
        for n in numpy.arange(min(neuron_index),max(neuron_index),range):
            commands.append('R')
            commands.append(str(n))
            commands.append(str(n+range-1))
        keys.append("x")

        # Remove spike file
        os.system("rm " + self.tm_path + 'train')      

        # Launch test mode
        self.configure_hardware(keys)

        # Read file
        try:
            data = numpy.genfromtxt(self.tm_path + "train")
        except:
            data = []

        if (data != []):

            # Neuron index
            neuron_list = []
            spikes_lists = []

            # Sort data
            for i in data:
                neuron_number = (-i[1]+7)*64 + i[2]

                if (not (neuron_number in neuron_list)):
                    neuron_list.append(int(neuron_number))

            # Organize spikes lists
            for i in neuron_list:
                spikes_lists.append([])

            for i in data:
                neuron_number = (-i[1]+7)*64 + i[2]
                spikes_lists[neuron_list.index(neuron_number)].append(i[0]*1e6)    

            # Calculate new neuron list
            neuron_list_new = []
            for i in neuron_list:
                if (i > -1 and i < 32):
                    neuron_list_new.append(i)
                if (i > 95 and i < 160):
                    neuron_list_new.append(i-96+32)
                if (i > 223 and i < 288):
                    neuron_list_new.append(i-224+32+64)
                if (i > 351 and i < 416):
                    neuron_list_new.append(i-352+32+128)
                if (i > 479 and i < 512):
                    neuron_list_new.append(i-480+32+128+64)

        return neuron_list_new, spikes_lists

    ## Calc ISI
    # @param spikes_list The input spikes list
    def calc_ISI(self,spikes_list):
        ISI = []
        for i in numpy.arange(1,len(spikes_list)):
            ISI.append(spikes_list[i]-spikes_list[i-1])

        return ISI

    ## Calc frequency
    # @param spikes_list The input spikes list
    def calc_freq(self,spikes_list):
        ISI = self.calc_ISI(spikes_list)
        mean_ISI = numpy.mean(ISI)
        if (mean_ISI != 0):
            return 1/mean_ISI
        else:
            return 0

    ## Calculate standard deviation of the spiking frequencies
    # @param spikes_list The input spikes list
    def calc_std(self,spikes_list):
        ISI = self.calc_ISI(spikes_list)
        freqs = []
        for i in ISI:
            if (i != 0):
                freqs.append(1/i)
        std_freq = numpy.std(freqs)
        return std_freq


