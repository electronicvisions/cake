## @package hardwareinterface
# Python interface of the BrainScaleS hardware

# vim: set noexpandtab:

# Import
import os
import numpy as np
import pylab
from scipy import optimize
import matplotlib.pyplot as plt
import sys
import time

## Class HardwareInterface
class HardwareInterface:

	## The constructor
	# @param hardware The desired hardware system. It can be "demonstrator", "WSS" or "USB"
	# @param scope_address The address of the scope
	# @param setup_address The ip address of the FPGA board
	def __init__(self,hardware='demonstrator',scope_address="129.206.176.32",setup_address="192.168.1.17"):

		self.hardware = hardware

		# Define max number of retry
		self.retry_max = 2

		if(hardware == 'WSS' or hardware == 'WSS_livedemo'):

			# Test mode path
			self.xmlfile_path = os.environ['SYMAP2IC_PATH'] + "/components/calibration/source/"

			# XML comfiguration file path
			self.tm_path = os.environ['SYMAP2IC_PATH'] + "/bin/"

			# Access mode, see ./test2 -h for more informations
			self.mode = "-bje2f"

			# Number of HICANNs in the chain
			self.h_chain = "8"

			# HICANN ID
			self.h_id = "0"

			setup_ip = (setup_address, "1700")

			tests2_args = ["-log", "0", "-r", "0"]

			# HW interface
			import demonstratorinterface as DEMO
			self.fgi = DEMO.DemonstratorInterface(self.tm_path,self.xmlfile_path,self.mode,self.h_chain,self.h_id,setup_ip,tests2_args)

			# Voltage conversion
			self.vCoeff = 2200

		if(hardware == 'USB' or hardware == 'WSS' or hardware == 'WSS_livedemo'):

			# Import ADC interface
			import adcinterface as adc
			self.adc = adc.ADCInterface()

		if(hardware == 'demonstrator' or hardware == 'livedemo' or hardware == 'no_scope' or hardware == 'USB'):
			# HICANN ID
			self.h_id = 0
			setup_ip = (setup_address, "1701")

			# HW interface
			from halbeinterface import HalbeInterface
			self.fgi = HalbeInterface(self.h_id, setup_ip)

			# Voltage conversion
			# self.vCoeff = 2200 # <- should be in calibration
			self.vCoeff = 1000

		if(hardware == 'demonstrator'):

			# Import scope
			import scopeinterface as scope

			# Scope parameters
			self.scope_address = scope_address
			self.channel = 1

			# Adjustment of scope channels
			self.scope_adjusted = False

			# Create scope interface
			self.scopi = scope.ScopeInterface(self.scope_address)

		# Save analog traces ?
		self.save_analog_traces = False

		# Default current
		self.current_default = 600 # DAC value

	## Main measurement function
	# @param hicann_index The index of HICANNs. For example : [4,5]
	# @param neuron_index The index of neurons. For example : [[1,2,3],[1,2,3]]
	# @param parameter The parameter to measure. Example : "EL"
	# @param parameters The neuron parameters
	# @param stimulus Value of the current stimulus
	# @param value The hardware value of the parameter that is currently being measured
	# @param spike_stimulus The list of spikes to send to the hardware
	def measure(self,hicann,neurons,parameter,parameters,stimulus=0,spike_stimulus=[]):

		if (self.hardware == 'USB' or self.hardware == 'WSS'):

			# Create main measurement array for all HICANNs
			measurement_array = []

			#if parameter in ['gL', 'digital_freq']:
			if parameter in ['digital_freq']:

				# Do digital measurement

				# Init HICANN
				self.fgi.init_L1()

				# Get frequencies for all neurons in current HICANN
				print 'Measuring frequencies ...'
				neurons, freqs = self.fgi.read_spikes_freq(neurons,range=32)

				measurement_array.append(freqs)

			else:

				# Init measurement array
				meas_array = []

				for n,current_neuron in enumerate(neurons):

					# Log
					print "Measuring neuron " + str(current_neuron)

					# Switch neurons
					#start_op = time.time()
					self.switch_neuron(current_neuron)
					#print 'Sweep time', time.time() - start_op

					#sys.exit()

					# Measure

					# Parameter specific measurement
					if (parameter == 'EL'):

						# Measure
						#start_op = time.time()
						measured_value = self.adc.get_mean() * self.vCoeff
						#print 'USB acq time', time.time() - start_op

						print "Measured value for EL : " + str(measured_value) + " mV"
						meas_array.append(measured_value)

					if (parameter == 'Vreset'):

						# Measure
						measured_value = self.adc.get_min() * self.vCoeff

						print "Measured value for Vreset : " + str(measured_value) + " mV"
						meas_array.append(measured_value)

					if (parameter == 'Vt'):

						# Measure
						measured_value = self.adc.get_max() * self.vCoeff

						print "Measured value for Vt : " + str(measured_value) + " mV"
						meas_array.append(measured_value)

					if (parameter == 'gL' or parameter == 'tauref' or parameter == 'a' or parameter == 'b' or parameter == 'Vexp'):

						# Measure
						measured_value = self.adc.get_freq()

						print "Measured frequency : " + str(measured_value) + " Hz"
						meas_array.append(measured_value)


					if (parameter == 'tw'):

						# Inject current
						self.fgi.set_stimulus(self.current_default)

						# Get scope trace
						t,v = self.adc.start_and_read_adc(400)

						# Apply fit
						tw = self.tw_fit(t,v,parameters['C'],parameters['gL'])

						# Append to meas array
						meas_array.append(tw)

						print "Measured value : " + str(tw)

					if (parameter == 'dT'):

						# Get scope trace
						t,v = self.adc.start_and_read_adc(400)

						# Calc dT
						dT = self.calc_dT(t,v,parameters['C'],parameters['gL'],parameters['EL'])

						# Append
						meas_array.append(dT)

						# Print result
						print "Measured value : " + str(dT)

					if (parameter == 'tausynx' or parameter=='tausyni'):

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

		if (self.hardware == 'demonstrator'):

			# Create main measurement array for all HICANNs
			measurement_array = []

			#if parameter in ["gL","tauref",'a','b','Vexp','digital_freq']:
			if parameter in ['a','b','Vexp','digital_freq']:
				# Do digital measurement

				for h, current_hicann in enumerate(hicann_index):

					# Init HICANN
					self.fgi.init_L1()

					# Get frequencies for all neurons in current HICANN
					print 'Measuring frequencies ...'
					neurons, freqs = self.fgi.read_spikes_freq(neuron_index[h],range=8)

					measurement_array.append(freqs)

			else:
				# Do analog measurement

				# Init scope
				self.scopi.scope_init(self.channel)
				# Init measurement array
				meas_array = []

				for n,current_neuron in enumerate(neurons):

					# Log
					print "Measuring neuron " + str(current_neuron)

					# Switch neurons
					#start_op = time.time()
					self.switch_neuron(current_neuron)
					#print 'Sweep time', time.time() - start_op

					# Measure

					# Retry counter
					retry = 0

					# Set scope channel and trigger
					self.scopi.set_trig(self.channel)

					# Parameter specific measurement
					if (parameter == 'EL'):

						# Measure
						try:
							measured_value = self.scopi.get_mean(self.channel)*self.vCoeff
						except:
							measured_value = 0

						print "Measured value for EL : " + str(measured_value) + " mV"
						meas_array.append(measured_value)

					if (parameter == 'Vreset'):

						# Adjusting scope
						if (self.scope_adjusted == False):
							while (retry < self.retry_max):
								try:
									retry += 1
									self.scopi.scope_adjust(self.channel)
									self.scope_adjusted = True
									break
								except:
									continue

						# Measure
						try:
							Vreset = self.scopi.get_min(self.channel)*self.vCoeff
						except:
							Vreset = 0

						# Try again if Vreset = 0
						if (Vreset == 0):
							try:
								Vreset = self.scopi.get_min(self.channel)*self.vCoeff
							except:
								Vreset = 0

						print "Measured value for Vreset : " + str(Vreset) + " mV"
						meas_array.append(Vreset)

					if (parameter == 'Vt'):

						# Adjusting scope
						if (self.scope_adjusted == False):
							while (retry < self.retry_max):
								try:
									retry += 1
									self.scopi.scope_adjust(self.channel)
									self.scope_adjusted = True
									break
								except:
									continue

						# Measure
						try:
							Vt = self.scopi.get_max(self.channel)*self.vCoeff
						except:
							Vt = 0

						# Try again if Vt = 0
						if (Vt == 0):
							try:
								Vt = self.scopi.get_max(self.channel)*self.vCoeff
							except:
								Vt = 0

						print "Measured value for Vt : " + str(Vt) + " mV"
						meas_array.append(Vt)

					if (parameter == 'tw'):

						# Inject current
						self.fgi.set_stimulus(self.current_default)

						# Adjusting scope
						self.scopi.scope_adjust(self.channel)

						# Get scope trace
						t,v = self.scopi.get_scope_value(self.channel)

						pylab.plot(t,v)
						pylab.show()

						# Apply fit
						tw = self.tw_fit(t,v,parameters['C'],parameters['gL'])

						# Append to meas array
						meas_array.append(tw)

						# Print result
						print "Measured value : " + str(tw)

					if (parameter == 'tausynx' or parameter=='tausyni'):

						# Activate BEG
						self.fgi.activate_BEG('ON','REGULAR',2000,current_neuron)
						print 'BEG active'

						# Get scope trace
						self.scopi.scope_adjust_syn(self.channel)
						time.sleep(1)
						t,v = self.scopi.scope_sta(20e-6,self.channel)

						# Convert v with scaling
						for i in range(len(v)):
							v[i] = v[i]*self.vCoeff
							t[i] = t[i]*1e6

						# if (self.save_analog_traces == True):
						# 	np.savetxt(parameter + str(value) + '_' + str(rep) + '_t.txt',t)
						# 	np.savetxt(parameter + str(value) + '_' + str(rep) + '_v.txt',v)

						if (parameter == 'tausynx'):
							fit = self.fit_psp(t,v,parameter,parameters[parameter],parameters['Esynx'])

						if (parameter == 'tausyni'):
							fit = self.fit_psp(t,v,parameter,parameters[parameter],parameters['Esyni'])

						print "Measured value : " + str(fit)
						meas_array.append(fit)

					if (parameter == 'dT'):

						# Adjust scope
						self.scopi.scope_adjust_syn(self.channel)

						# Get scope trace
						t,v = self.scopi.get_scope_value(self.channel)

						# Calc dT
						dT = self.calc_dT(t,v,parameters['C'],parameters['gL'],parameters['EL'])

						# Append
						meas_array.append(dT)

						# Print result
						print "Measured value : " + str(dT)

					if (parameter == 'gL' or parameter == 'tauref' or parameter == 'a' or parameter == 'b' or parameter == 'Vexp' or parameter == 'analog_freq'):

						# Adjusting scope
						if (self.scope_adjusted == False):
							while (retry < self.retry_max):
								try:
									retry += 1
									self.scopi.scope_adjust(self.channel)
									self.scopi.set_time_div(10e-6)
									self.scope_adjusted = True
									break
								except:
									continue
							time.sleep(0.5)

						# Get frequency
						try:
							freq = self.scopi.get_avg_freq(self.channel)
						except:
							freq = 0

						print "Measured value : " + str(freq) + " Hz"
						meas_array.append(freq)

					if (parameter == 'analog_freq_current'):

						# Apply stimulus and sweep
						self.fgi.set_constant_stimulus(stimulus)
						self.fgi.sweep_neuron(current_neuron,current=1)

						# Adjusting scope
						if (self.scope_adjusted == False):
							while (retry < self.retry_max):
								try:
									retry += 1
									self.scopi.scope_adjust(self.channel)
									self.scope_adjusted = True
									break
								except:
									continue
							time.sleep(0.5)

						# Get frequency
						try:
							freq = self.scopi.get_avg_freq(self.channel)
						except:
							freq = 0

						print "Measured value : " + str(freq) + " Hz"
						meas_array.append(freq)

					if (parameter == 'Trace'):

						# Apply stimulus and sweep
						self.fgi.set_stimulus(stimulus)
						self.fgi.sweep_neuron(current_neuron,current=1)

						if (spike_stimulus != []):
							print 'Activate BEG ...'
							self.fgi.activate_BEG('ON',spike_stimulus[0],spike_stimulus[1],current_neuron)

						# Measure trace
						self.scopi.scope_adjust(self.channel)
						t,v = self.scopi.get_scope_value(self.channel)

						for i in range(len(v)):
							v[i] = v[i]*self.vCoeff
							t[i] = t[i]*1e6

						meas_array.append([t,v])

					if (parameter == 'freq_noscope'):

						# Apply stimulus
						self.fgi.set_stimulus(stimulus)

						# Switch neurons
						if(current_neuron < 256):
							self.fgi.sweep_neuron(current_neuron,'top')
						else:
							self.fgi.sweep_neuron(current_neuron-256,'bottom')

						# Get frequency
						try:
							freq = self.scopi.get_avg_freq(self.channel)
						except:
							freq = 0

						print "Measured frequency : " + str(freq) + " Hz"
						#sys.exit()
						meas_array.append(freq)

				#self.fgi.disable_aout()

				measurement_array = meas_array

				# Reset scope status
				self.scope_adjusted = False

		return measurement_array

	## Main configuration function
	# @param hicann_index The index of HICANNs. For example : [4,5]
	# @param neuron_index The index of neurons. For example : [[1,2,3],[1,2,3]]
	# @param parameter The parameter to measure. Example : "EL"
	# @param option The configuration option.
	def configure(self,hicann,neurons,parameters,option='configure'):
		if self.hardware in ('WSS', 'USB', 'demonstrator', 'no_scope', 'livedemo'):
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

		#print tfit
		#print dT*1e3

		trace_fit = self.fit_exp_trace(tfit, voltage)
		#self.plot(v, trace, trace_fit)

		return dT*1e3

	## Calc dT from voltage trace
	# @param t Time array
	# @param v Voltage array
	# @param C Capacitance of the neuron
	# @param gL Membrane leakage conductance of the neuron
	# @param EL Membrane resting potential of the neuron
	def calc_dT(self,t,v,C,gL,EL):

		# Smooth voltages
		#v = self.simi.smooth(v,7)

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

		#print end_found

		#end = exp.index(max(exp))

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

		#pylab.plot(v,exp)
		#pylab.show()
		#sys.exit()

		return self.exp_fit(v,exp)

	## Switch to the correct hardware neuron
	# current_neuron The number of the desired neuron
	def switch_neuron(self,current_neuron):

		if(current_neuron < 128):
			self.fgi.sweep_neuron(current_neuron,'top')
			side = 'left'
		if(current_neuron > 127 and current_neuron < 256):
			self.fgi.sweep_neuron(-current_neuron + 383,'top')
			side = 'right'
		if(current_neuron > 255 and current_neuron < 384):
			self.fgi.sweep_neuron(current_neuron,'bottom')
			side = 'left'
		if(current_neuron > 383):
			self.fgi.sweep_neuron(-current_neuron + 895,'bottom')
			side = 'right'
