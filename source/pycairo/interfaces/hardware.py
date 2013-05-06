'''Wrapper of multiple hardware variants (wafer/vertical setup)'''

import numpy as np
from scipy import optimize

import pycairo.interfaces.adc
from pycairo.interfaces.halbe import HalbeInterface
import pycairo.logic.helpers
import pycairo.config.hardware as config

class HardwareInterface:
    # @param hardware The desired hardware system. It can be "WSS" or "USB"
    # @param setup_address The ip address of the FPGA board
    def __init__(self, hardware, setup_address):
        '''Select Wafer Scale System (WSS) or vertical setup with ADC board via USB (USB)

        Args:
            hardware: 'USB' for vertical setup with ADC board or 'WSS' for wafer
            setup_address: FPGA IP(v4)
        '''

        if hardware not in ('WSS', 'USB'):
            raise Exception, 'non-ADC readout not supported anymore'

        self.adc = pycairo.interfaces.adc.ADCInterface()

        if hardware  == 'USB':
            # HICANN ID
            self.h_id = 0
            setup_ip = (setup_address, "1701")

        # HW interface
        self.halbe_if = HalbeInterface(self.h_id, setup_ip)

        self.helpers = pycairo.logic.helpers.Helpers()

    def measure(self, hicann, neurons, parameter, parameters, stimulus=0, spike_stimulus=[]):
        '''Main measurement function.

        Args:
            hicann_index The index of HICANNs. For example : [4,5]
            neuron_index The index of neurons. For example : [[1,2,3],[1,2,3]]
            parameter The parameter to measure. Example : "EL"
            parameters The neuron parameters
            stimulus Value of the current stimulus
            value The hardware value of the parameter that is currently being measured
            spike_stimulus The list of spikes to send to the hardware
        '''

        measurement_array = [] # main measurement array for all HICANNs

        #if parameter in ['gL', 'digital_freq']:
        if parameter in ['digital_freq']: # do digital measurement
            self.halbe_if.init_L1() # Init HICANN

            # Get frequencies for all neurons in current HICANN
            print 'Measuring frequencies ...'
            neurons, freqs = self.halbe_if.read_spikes_freq(neurons,range=32)

            measurement_array.append(freqs)
        else:
            meas_array = [] # Init measurement array
            for n,current_neuron in enumerate(neurons):
                print "Measuring neuron " + str(current_neuron)
                self.switch_neuron(current_neuron)

                # Parameter specific measurement
                if (parameter == 'EL'):
                    measured_value = self.adc.get_mean() * config.vCoeff
                    print "Measured value for EL : " + str(measured_value) + " mV"
                    meas_array.append(measured_value)
                elif (parameter == 'Vreset'):
                    measured_value = self.adc.get_min() * config.vCoeff
                    print "Measured value for Vreset : " + str(measured_value) + " mV"
                    meas_array.append(measured_value)
                elif (parameter == 'Vt'):
                    measured_value = self.adc.get_max() * config.vCoeff
                    print "Measured value for Vt : " + str(measured_value) + " mV"
                    meas_array.append(measured_value)
                elif (parameter == 'gL' or parameter == 'tauref' or parameter == 'a' or parameter == 'b' or parameter == 'Vexp'):
                    measured_value = self.adc.get_freq()
                    print "Measured frequency : " + str(measured_value) + " Hz"
                    meas_array.append(measured_value)
                elif (parameter == 'tw'):
                    # Inject current
                    self.halbe_if.set_stimulus(config.current_default)

                    # Get trace
                    trace = self.adc.read_adc(config.sample_time_tw)
                    t,v = trace.time, trace.voltage

                    # Apply fit
                    tw = self.helpers.tw_fit(t,v,parameters['C'],parameters['gL'])

                    print "Measured value : " + str(tw)
                    meas_array.append(tw)
                elif (parameter == 'dT'):
                    # Get trace
                    trace = self.adc.read_adc(config.sample_time_dT)
                    t,v = trace.time, trace.voltage

                    # Calc dT
                    dT = self.helpers.calc_dT(t,v,parameters['C'],parameters['gL'],parameters['EL'])

                    print "Measured value : " + str(dT)
                    meas_array.append(dT)
                elif (parameter == 'tausynx' or parameter=='tausyni'):
                    # Activate BEG
                    self.halbe_if.activate_BEG('ON','REGULAR',2000,current_neuron)
                    print 'BEG active'

                    # Get trace
                    t,v = self.adc.adc_sta(20e-6)

                    # Convert v with scaling
                    for i in range(len(v)):
                        v[i] = v[i]*self.config.vCoeff/1000
                        t[i] = t[i]*1000

                    if (parameter == 'tausynx'):
                        fit = self.helpers.fit_PSP(t,v,parameter,parameters[parameter],parameters['Esynx'])
                    if (parameter == 'tausyni'):
                        fit = self.helpers.fit_PSP(t,v,parameter,parameters[parameter],parameters['Esyni'])

                    print "Measured value : " + str(fit)
                    meas_array.append(fit)
            measurement_array = meas_array
        return measurement_array

    def configure(self, hicann, neurons, parameters, option='configure'):
        """Main configuration function. Sends config to HALbe interface.

        Args:
            hicann_index The index of HICANNs. For example : [4,5]
            neuron_index The index of neurons. For example : [[1,2,3],[1,2,3]]
            parameter The parameter to measure. Example : "EL"
            option The configuration option.
        """

        self.halbe_if.send_fg_configure(neurons, parameters, option)

    def switch_neuron(self,current_neuron):
    	'''Switch to the correct hardware neuron.

		Args:
    		current_neuron: number of the desired neuron.'''

        if(current_neuron < 128):
            self.halbe_if.sweep_neuron(current_neuron,'top')
            side = 'left'
        elif(current_neuron > 127 and current_neuron < 256):
            self.halbe_if.sweep_neuron(-current_neuron + 383,'top')
            side = 'right'
        elif(current_neuron > 255 and current_neuron < 384):
            self.halbe_if.sweep_neuron(current_neuron,'bottom')
            side = 'left'
        elif(current_neuron > 383):
            self.halbe_if.sweep_neuron(-current_neuron + 895,'bottom')
            side = 'right'
