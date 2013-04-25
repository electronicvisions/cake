'''Wrapper of multiple hardware variants (wafer/vertical setup)'''

import os
import numpy as np
import pylab
from scipy import optimize
import matplotlib.pyplot as plt
import sys
import time

class HardwareInterface:
    # @param hardware The desired hardware system. It can be "WSS" or "USB"
    # @param setup_address The ip address of the FPGA board
    def __init__(self, hardware, setup_address):
        '''Select Wafer Scale System (WSS) or vertical setup with ADC board via USB (USB)

        Args:
            hardware: 'USB' for vertical setup with ADC board or 'WSS' for wafer
            setup_address: FPGA IP(v4)
        '''

        self.hardware = hardware

        # Define max number of retry
        self.retry_max = 2

        if hardware in ['USB', 'WSS']:
            # Import ADC interface
            from pycairo.interfaces import adc
            self.adc = adc.ADCInterface()

        if hardware  == 'USB':
            # HICANN ID
            self.h_id = 0
            setup_ip = (setup_address, "1701")

            # HW interface
            from pycairo.interfaces.halbe import HalbeInterface
            self.fgi = HalbeInterface(self.h_id, setup_ip)

            # Voltage conversion
            self.vCoeff = 1000 # FIXME: should be in calibration

        self.save_analog_traces = False

        self.current_default = 600 # Default current, DAC value

    def measure(self,hicann,neurons,parameter,parameters,stimulus=0,spike_stimulus=[]):
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

        if self.hardware in ['USB', 'WSS']:

            # Create main measurement array for all HICANNs
            measurement_array = []

            #if parameter in ['gL', 'digital_freq']:
            if parameter in ['digital_freq']: # do digital measurement
                self.fgi.init_L1() # Init HICANN

                # Get frequencies for all neurons in current HICANN
                print 'Measuring frequencies ...'
                neurons, freqs = self.fgi.read_spikes_freq(neurons,range=32)

                measurement_array.append(freqs)

            else:
                meas_array = [] # Init measurement array
                for n,current_neuron in enumerate(neurons):
                    print "Measuring neuron " + str(current_neuron)
                    self.switch_neuron(current_neuron)

                    # Parameter specific measurement
                    if (parameter == 'EL'):

                        # Measure
                        #start_op = time.time()
                        measured_value = self.adc.get_mean() * self.vCoeff
                        #print 'USB acq time', time.time() - start_op

                        print "Measured value for EL : " + str(measured_value) + " mV"
                        meas_array.append(measured_value)

                    elif (parameter == 'Vreset'):
                        measured_value = self.adc.get_min() * self.vCoeff

                        print "Measured value for Vreset : " + str(measured_value) + " mV"
                        meas_array.append(measured_value)

                    elif (parameter == 'Vt'):
                        measured_value = self.adc.get_max() * self.vCoeff

                        print "Measured value for Vt : " + str(measured_value) + " mV"
                        meas_array.append(measured_value)

                    elif (parameter == 'gL' or parameter == 'tauref' or parameter == 'a' or parameter == 'b' or parameter == 'Vexp'):
                        measured_value = self.adc.get_freq()

                        print "Measured frequency : " + str(measured_value) + " Hz"
                        meas_array.append(measured_value)

                    elif (parameter == 'tw'):
                        # Inject current
                        self.fgi.set_stimulus(self.current_default)

                        # Get trace
                        t,v = self.adc.start_and_read_adc(400)

                        # Apply fit
                        tw = self.tw_fit(t,v,parameters['C'],parameters['gL'])

                        print "Measured value : " + str(tw)
                        meas_array.append(tw)

                    elif (parameter == 'dT'):
                        # Get trace
                        t,v = self.adc.start_and_read_adc(400)

                        # Calc dT
                        dT = self.calc_dT(t,v,parameters['C'],parameters['gL'],parameters['EL'])

                        print "Measured value : " + str(dT)
                        meas_array.append(dT)

                    elif (parameter == 'tausynx' or parameter=='tausyni'):
                        # Activate BEG
                        self.fgi.activate_BEG('ON','REGULAR',2000,current_neuron)
                        print 'BEG active'

                        # Get trace
                        t,v = self.adc.adc_sta(20e-6)

                        # Convert v with scaling
                        for i in range(len(v)):
                            v[i] = v[i]*self.vCoeff/1000
                            t[i] = t[i]*1000

                        if (parameter == 'tausynx'):
                            fit = self.fit_PSP(t,v,parameter,parameters[parameter],parameters['Esynx'])

                        if (parameter == 'tausyni'):
                            fit = self.fit_PSP(t,v,parameter,parameters[parameter],parameters['Esyni'])

                        print "Measured value : " + str(fit)
                        meas_array.append(fit)

            measurement_array = meas_array
        else:
            raise Exception, 'non-ADC readout not supported anymore'

        return measurement_array

    ## Main configuration function
    # @param hicann_index The index of HICANNs. For example : [4,5]
    # @param neuron_index The index of neurons. For example : [[1,2,3],[1,2,3]]
    # @param parameter The parameter to measure. Example : "EL"
    # @param option The configuration option.
    def configure(self,hicann,neurons,parameters,option='configure'):
        if self.hardware in ('WSS', 'USB'):
            self.fgi.send_fg_configure(neurons, parameters, option)

########### Helpers functions ##############

    ## Fit PSP
    # @param tm Time array
    # @param psp Voltage array
    # @param parameter Parameter to be fitted, can be 'tausynx' or 'tausyni'
    # @param Esyn Synaptic reversal potential
    # @param Cm Capacitance of the neuron
    def fit_psp(self,tm,psp,parameter,Esyn,Cm):
        # Seperate between exc and inh
        if (parameter == 'tausynx'):

            # Measure base
            base = min(psp)

            # Initial fit params (see pfit below)
            p0 = [2, 1, base, 2]

            # Calc error function
            errfunc = lambda p, t, y: self.psp_trace_exc(p, t) - y

            pfit, success = optimize.leastsq(errfunc, p0[:], args=(tm, psp))
            tau_syn, tau_eff, v_offset, t_offset = pfit

            trace_fit = self.psp_trace_exc(pfit, tm)

        if (parameter == 'tausyni'):

            # Measure base
            base = max(psp)

            # Initial fit params (see pfit below)
            p0 = [2, 1, base, 2]

            # Calc error function
            errfunc = lambda p, t, y: self.psp_trace_inh(p, t) - y

            pfit, success = optimize.leastsq(errfunc, p0[:], args=(tm, psp))
            tau_syn, tau_eff, v_offset, t_offset = pfit

            trace_fit = self.psp_trace_inh(pfit, tm)
        #self.plot(tm, psp, trace_fit)

        # Weight calculation
        e_rev_E = Esyn
        # nF
        Cm = Cm
        v_rest = base
        calc_weights = 40*Cm*(1./tau_eff - 1./tau_syn)/(e_rev_E - v_rest)

        return pfit[0]

    ## Fit PSP function for exc PSPs
    # @param p Array with the parameters of the fit
    # @param t Time array
    def psp_trace_exc(self, p, t):
        return self.theta(t - p[3])*40*(np.exp(-(t-p[3])/p[0]) - np.exp(-(t-p[3])/p[1])) + p[2]

    ## Fit PSP function for inh PSPs
    # @param p Array with the parameters of the fit
    # @param t Time array
    def psp_trace_inh(self, p, t):
        return self.theta(t - p[3])*-40*(np.exp(-(t-p[3])/p[0]) - np.exp(-(t-p[3])/p[1])) + p[2]

    ## Fit PSP
    # @param t Time array
    def theta(self,t):
        return (t>0)*1.

    ## Fit tw trace
    # @param p Array with the parameters of the fit
    # @param t Time array
    def fit_tw_trace(self, p, t):
        return (p[2]*np.cos(p[1]*t)+p[3]*np.sin(p[1]*t))*np.exp(-(p[0]*t)) + p[4]

    ## Fit tw trace
    # @param tm Time array
    # @param trace Voltage array
    # @param C Capacitance of the neuron
    # @param gL Membrane leakage conductance of the neuron
    def tw_fit(self,tm,trace, C, gL):
        # Initial fit params (see pfit below)
        # l, w, A, B, v_inf, dt
        p0 = [300000,400000,-0.02, 0.15, 0.63]

        # Fit the data
        errfunc = lambda p, t, y: self.fit_tw_trace(p, t) - y
        tfit, success = optimize.leastsq(errfunc, p0[:], args=(tm, trace))
        l, w, A, B, v_inf = tfit

        trace_fit = self.fit_tw_trace(tfit, tm)
        #self.plot(tm, trace, trace_fit)

        # Calc tw
        tm = C*1e-12/(gL*1e-9)
        tw = tm/(2*tfit[0]*tm-1)

        return tw*1e6

    ## Plot the trace vs the fitted trace
    # @param tm Time array
    # @param trace Voltage array
    # @param trace_fit Fitted voltage array
    def plot(self, tm, trace, trace_fit):
        pylab.plot(tm, trace, c='k')
        pylab.plot(tm, trace_fit, c='r')
        pylab.grid(True)
        pylab.show()

    ## Fit trace to find dT
    # @param p Array of parameter for the fit function
    # @param t Time array
    def fit_exp_trace(self, p, t):
        return p[0]*np.exp(t/p[1])

    ## Fit trace to find dT
    # @param voltage Voltage array
    # @param current Current array
    def exp_fit(self,voltage,current):
        # Initial fit params (see pfit below)
        # l, w, A, B, v_inf, dt
        p0 = [0.2e-10,12e-3]

        # Fit the data
        errfunc = lambda p, t, y: self.fit_exp_trace(p, t) - y
        tfit, success = optimize.leastsq(errfunc, p0[:], args=(voltage, current))

        i0,dT = tfit

        trace_fit = self.fit_exp_trace(tfit, voltage)

        return dT*1e3

    ## Calc dT from voltage trace
    # @param t Time array
    # @param v Voltage array
    # @param C Capacitance of the neuron
    # @param gL Membrane leakage conductance of the neuron
    # @param EL Membrane resting potential of the neuron
    def calc_dT(self,t,v,C,gL,EL):
        # Calculate exp current
        diff = []
        for i in range(1,len(v)):
            diff.append((v[i]-v[i-1])/(t[i]-t[i-1]))

        exp = []
        for i in range(len(diff)):
            exp.append(C*1e-12/(gL*1e-9)*diff[i] + (v[i]-EL*1e-3))

        # Cut the right part
        end_found = False
        end = 0
        for i in range(1,len(exp)):
            if (((exp[i] - exp[i-1]) > 2) and end_found==False):
                end_found = True
                end = i-10

        v = v[:end]
        exp = exp[:end]
        t = t[:end]

        # Cut the left part
        new_exp = []
        new_v = []
        new_t = []

        for i in range(len(exp)):
            if (exp[i] > 0.015):
                new_exp.append(exp[i])
                new_v.append(v[i])
                new_t.append(t[i])

        v = new_v
        exp = new_exp
        t = new_t

        return self.exp_fit(v,exp)

    def switch_neuron(self,current_neuron):
    	'''Switch to the correct hardware neuron.

		Args:
    		current_neuron: number of the desired neuron.'''

        if(current_neuron < 128):
            self.fgi.sweep_neuron(current_neuron,'top')
            side = 'left'
        elif(current_neuron > 127 and current_neuron < 256):
            self.fgi.sweep_neuron(-current_neuron + 383,'top')
            side = 'right'
        elif(current_neuron > 255 and current_neuron < 384):
            self.fgi.sweep_neuron(current_neuron,'bottom')
            side = 'left'
        elif(current_neuron > 383):
            self.fgi.sweep_neuron(-current_neuron + 895,'bottom')
            side = 'right'
