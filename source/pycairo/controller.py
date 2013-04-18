'''Controller for the calibration of the BrainScaleS hardware'''

from . import simulator as sim
import time
import numpy as np
import sys
import os

class CalibrationController:
	'''Main calibration class'''

	## Creation of the different interfaces to devices used for calibration.
	# @param hardware The type of hardware to use. Possible choices are demonstrator, USB, WSS, scope_only
	# @param scope_address The IP address of the scope
	# @param neurons_range The range of neurons to calibrate per chip
	# @param fpga The logical number of the FPGA board to use
	def __init__ (self,hardware='demonstrator',scope_address = "129.206.176.32",neurons_range=[],fpga=6):

		self.hardware = hardware
		self.neurons_range = neurons_range
		self.fpga = fpga

		if hardware in ('WSS', 'USB', 'demonstrator', 'db_only'):
			# DB interface creation
			import databaseinterface as DB
			self.dbi = DB.DatabaseInterface()
			# Create DB if not already existing
			if self.dbi.is_empty() == True:
				self.dbi.create_db(hardware)
			setup_address = '192.168.1.' + str(self.dbi.get_fpga_ip(fpga))
		if hardware in ('demonstrator', 'WSS', 'USB'):

			# Create HW interface
			import hardwareinterface as HWI
			self.hwi = HWI.HardwareInterface(hardware,scope_address,setup_address)

			# Scaled to hardware module
			import scaledhw as scali
			self.scali = scali.scaledHW()

		if hardware == 'scope_only':

			# Import scope
			import scopeinterface as scope
			# Scope parameters
			self.scope_address = scope_address
			self.channeltype = "C"
			# Create scope interface
			self.scopi = scope.ScopeInterface(self.scope_address)

		# Import in all cases
		self.simi = sim.Simulator()

		# Global commands
		self.debug_mode = False

		# Calibration data for each parameter : min, max, number of values, number of measurements
		self.parameter_ranges = {
				"EL": {"min": 300, "max": 800, "pts": 6, "reps": 3},
				"Vreset": {"min": 400, "max": 600, "pts": 3, "reps": 2},
				"Vt": {"min": 600, "max": 800, "pts": 3, "reps": 2},
				"gL": {"min": 100, "max": 600, "pts": 4, "reps": 4},
				"tauref": {"min": 10, "max": 100, "pts": 4, "reps": 4},
				"dT": {"min": 50, "max": 400, "pts": 3, "reps": 3},
				"Vexp": {"min": 100, "max": 400, "pts": 3, "reps": 3},
				"b": {"min": 50, "max": 2000, "pts": 3, "reps": 3},
				"tausynx": {"min": 1300, "max": 1500, "pts": 4, "reps": 4},
				"tausyni": {"min": 1300, "max": 1500, "pts": 3, "reps": 2},
				"tw": {"min": 50, "max": 2000, "pts": 3, "reps": 3},
				"a": {"min": 50, "max": 400, "pts": 5, "reps": 5}
			}

		# Calibration default values
		self.Vt_default_max = 1700 # mV
		self.EL_default_max = 1100 # mV
		self.gL_default = 1000 # nA
		self.EL_default = 600 # mV
		self.a_default = 2000 # nA
		self.tauw_default = 1000 # nA

		self.EL_IF_value = 800 # mV
		self.Vreset_IF_value = 500 # mV
		self.Vt_IF_value = 700 # mV
		self.gL_IF_value = 1000 # nA
		self.all_IF_values = {
				"Vreset" : self.Vreset_IF_value,
				"EL" : self.EL_IF_value,
				"Vt" : self.Vt_IF_value,
				"gL" : self.gL_IF_value }

		self.Vexp_default = 650 # mV
		self.dT_default = 10 # mV

		self.Esynx_default = 1300 # mV
		self.Esyni_default = 200 # mV

		self.current_default = 200 # nA


# ***************************************************************************
#							Calibration function
# ***************************************************************************

	def calibrate(self, model='LIF'):
		"""Main calibration function
		@param model The model can be chosen between "LIF", "ALIF", "AdEx" and single_param, or directly the name of the parameter like "EL"
		"""
		# Log
		print "### Calibration software started ###"
		print ""
		calib_start = time.time()

		# Parameters per model
		LIF_parameters = ['EL','Vreset','Vt','gL','tauref','tausynx','tausyni']
		ALIF_parameters = LIF_parameters + ['a','tw','b']
		AdEx_parameters = ALIF_parameters + ['dT','Vexp']

		parameter_set = {
				'single_param' : ['EL'],
				'LIF'  : LIF_parameters,
				'ALIF' : ALIF_parameters,
				'AdEx' : AdEx_parameters,
				}

		if isinstance(model, basestring):
			parameters = parameter_set[model]
		else:
			parameters = model

		# Creation of neuron index

		for param in parameters:
			hicann_index, neuron_index = self.dbi.create_neuron_index(param,self.neurons_range,self.fpga)
			print "Reseting Hardware"
			for ii, hicann in enumerate(hicann_index):
				self.calibrate_parameter(param, hicann, neuron_index[ii])

		print "### Calibration completed in " + str(time.time() - calib_start) + " s ###"

	## Launch calibration for one parameter
	# @param parameter The parameter to be calibrated, for example "EL"
	# @param hicann HICANN to be calibrated
	# @param neurons list of neurons to be calibrated
	def calibrate_parameter(self, parameter, hicann, neurons):

		# Log
		results_print = True
		start_time = time.time()
		if neurons:
			print "## Starting calibration of parameter {} on HICANN {}".format(parameter, hicann)
			print "Calibrating neurons " + ", ".join([str(jj) for jj in neurons])
		else:
			print "## Parameter {} on HICANN {} allready calibrated for neurons".format(
					parameter, hicann)
			return

		####### Initialize #######

		# Log
		print "Init phase started"

		# Track calibration status
		self.n = len(neurons)
		start = time.time()

		parameters = self.get_parameters(parameter)

		# Create input_array
		input_array = self.get_steps(parameter)
		repetitions = self.parameter_ranges[parameter]['reps']

		output_array_mean = []
		output_array_err = []

		print "Init phase finished in " + str(time.time()-start_time) + " s"

		####### Measure #######

		start_measure = time.time()
		print "Measurement phase started"

		# Main measurement loop
		for index, value in enumerate(input_array):

			print "Measure for value " + str(value) + " [" + str(len(input_array) - index) + " values remaining]"

			# Create measurement arrays for mean values and errors

			# Set value to be calibrated
			parameters[parameter] = value

			# Convert parameters for one HICANN
			try:
				#Use calibration database
				calibrated_parameters = self.scali.scaled_to_hw(hicann,neurons,parameters,parameters=parameter+'_calibration')
			except:
				# If no calibration available, use direct translation
				calibrated_parameters = self.scali.scaled_to_hw(hicann,neurons,parameters,parameters='None')

			meas_array = []
			# Do a given number of repetitions
			for run in range(repetitions):
				print "Starting repetition {} of {}".format(run+1, repetitions)
				result = self.meassure(hicann, neurons, parameter, parameters, value, calibrated_parameters)
				meas_array.append(result)

			meas_array = np.array(meas_array)
			meas_array_mean = np.mean(meas_array.T, axis=1)
			meas_array_err = np.std(meas_array.T, axis=1)

			output_array_mean.append(meas_array_mean)
			output_array_err.append(meas_array_err)
			scope_adjusted = False


		# Log
		print "Measurement phase completed in " + str(time.time() - start_measure) + " s"

		####### Process #######

		# Log
		print "Process phase started"
		process_start = time.time()

		# Sort measurements
		sorted_array_mean = np.array(output_array_mean).T
		sorted_array_err  = np.array(output_array_err).T

		# Print for debug
		if results_print == True:
			print "Sorted array, mean :", sorted_array_mean
			print ""
			print "Sorted array, error :", sorted_array_err
			print ""

		processed_array_mean, processed_array_err = self.process_result(
				parameter, parameters, sorted_array_mean, sorted_array_err)
		processed_array_mean = np.array(processed_array_mean)
		processed_array_err = np.array(processed_array_err)

		# Print results for debug
		if (results_print == True):
			print "Processed array, mean :", processed_array_mean
			print ""
			print "Processed array, error :", processed_array_err

		# Log
		print "Process phase completed in " + str(time.time() - process_start) + " s"

		####### Store #######

		# Log
		print "Store phase started"
		store_start = time.time()

		db_input_array = input_array
		if parameter in ('tauref', 'tw'):
			db_input_array = 1.0/input_array

		for n,neuron in enumerate(neurons):

			# Compute calibration function
			try:
				parameter_fit = self.compute_calib_function(processed_array_mean[n], db_input_array)

				# Set the parameter as calibrated
				self.dbi.change_parameter_neuron(hicann, neuron, parameter+"_calibrated", True)
			except Exception as e: #TODO catch specific exception
				# If fail, don't store anything, mark as non calibrated
				parameter_fit = []
				self.dbi.change_parameter_neuron(hicann,neuron,parameter+"_calibrated",False)
				raise e

			# Store function in DB
			self.dbi.change_parameter_neuron(hicann,neuron,parameter+"_fit",parameter_fit)

			# Store standard deviation in DB
			self.dbi.change_parameter_neuron(hicann,neuron,parameter+"_dev", processed_array_err[n].tolist())

			# Store data in DB
			self.dbi.change_parameter_neuron(hicann,neuron,parameter,[processed_array_mean[n].tolist(),db_input_array.tolist()])

		# Evaluate calibration
		# self.dbi.evaluate_db(hicann,neuron_index[h],parameter)

		# Set HICANN as calibrated ?
		self.dbi.check_hicann_calibration(hicann)

		# Log
		print "Store phase completed in " + str(time.time() - store_start) + " s"
		print ""
		print "## Calibration of parameter " + parameter + " completed in " + str(time.time() - start_time) + " s ##"
		print "## Calibration took " + str((time.time() - start_time) / len(neurons)) + " s per neuron ##"
		print ""

# ***************************************************************************
#							Other functions
# ***************************************************************************
	def get_parameters(self, parameter):
		parameters = {}
		# Parameter specific initialization
		if parameter == 'EL':
			parameters.update( {'Vt' : self.Vt_default_max, "gL" : self.gL_default } )

		if parameter in ('Vreset', 'Vt'):
			parameters.update( {'EL' : self.EL_default_max, "gL" : self.gL_default } )

		if parameter == 'gL':
			parameters.update( {"Vreset" : self.Vreset_IF_value, "EL" : self.EL_IF_value,
				"Vt" : self.Vt_IF_value } )

		if parameter == 'tauref':
			parameters.update( self.all_IF_values )

		if parameter == 'a':
			parameters.update( self.all_IF_values )
			parameters["tauw"] = self.tauw_default

		if parameter == 'tw':
			parameters.update( self.all_IF_values )
			parameters["a"] = self.a_default

		if parameter == 'b':
			parameters.update( self.all_IF_values )
			parameters.update( {"a" : self.a_default, "tauw" : self.tauw_default } )

		if parameter in ('tausynx', 'tausyni'):
			parameters.update( self.all_IF_values )
			parameters["EL"] = self.EL_default
			parameters["Vt"] = self.Vt_default_max
			parameters["gL"] = 1000
			parameters["Esynx"] = self.Esynx_default
			parameters["Esyni"] = self.Esyni_default

		if parameter == 'dT':
			parameters.update( self.all_IF_values )
			parameters["expAct"] = 2000
			parameters["Vexp"] = self.Vexp_default

		if parameter == 'Vexp':
			parameters.update( self.all_IF_values )
			parameters["expAct"] = 2000
			parameters["dT"] = self.dT_default
		return parameters

	def process_result(self, parameter, parameters, sorted_array_mean, sorted_array_err):
		print
		if parameter in ('EL', 'Vt', 'Vreset', 'tw', 'dT'):
			return sorted_array_mean, sorted_array_err

		if parameter == 'gL':
			# Calculate relation between frequency and gL
			gL_coefs = self.simi.get_gl_freq(500,1200,parameters['EL'],parameters['Vreset'],parameters['Vt'])
			mean = [self.poly_process(a, gL_coefs) for a in sorted_array_mean]
			err =  [self.poly_process(a, gL_coefs) for a in sorted_array_err]
			return mean, err

		if parameter == 'tauref':
			# Compute base frequency (tau_ref -> 0 )
			base_freq = self.simi.compute_freq(30e-6,0,parameters)
			calc_freq = lambda x: (1.0/x - 1.0/base_freq) * 1e6

			print base_freq

			m = sorted_array_mean[sorted_array_mean != 0.0]
			e = sorted_array_err[sorted_array_err != 0.0]
			return calc_freq(m), calc_freq(e)

		if parameter in ('tausynx', 'tausyni'):

			# Calculate relation between PSP integral and tausyn
			#tausyn_coefs = self.simi.getTausynPSP(1,10,parameters['EL'],parameters['Vreset'],parameters['Vt'], parameters['gL'],parameters['Esynx'])

			#for h, hicann in enumerate(hicann_index):

				#hicann_array_mean = []
			 	#hicann_array_err = []

			 	#for n, neuron in enumerate(neuron_index[h]):

			 		# Calcultate tausyn and error
			 		#hicann_array_mean.append(self.poly_process(sorted_array_mean[h][n],tausyn_coefs))
			 		#hicann_array_err.append(self.poly_process(sorted_array_err[h][n],tausyn_coefs))

			 	#processed_array_mean.append(hicann_array_mean)
			 	#processed_array_err.append(hicann_array_err)

			# Fitting is already done in HW interface, just transmit results
			processed_array_err = sorted_array_err
			processed_array_mean = sorted_array_mean

			print processed_array_err
			print processed_array_mean
			return sorted_array_mean, sorted_array_err

		assert False, "TODO, fix stuff for other parameters"
		if (parameter == 'a'):

			# Calculate relation between frequency and gL
			a_coefs = self.simi.get_a_freq(200,600,parameters['EL'],parameters['Vreset'],parameters['Vt'],parameters['gL'])

			for h, hicann in enumerate(hicann_index):

				hicann_array_mean = []
				hicann_array_err = []

				for n, neuron in enumerate(neuron_index[h]):

					# Calculate a and error
					hicann_array_mean.append(self.poly_process(sorted_array_mean[h][n],a_coefs))
					hicann_array_err.append(self.poly_process(sorted_array_err[h][n],a_coefs))

				processed_array_mean.append(hicann_array_mean)
				processed_array_err.append(hicann_array_err)

		if (parameter == 'b'):

			# Calculate relation between frequency and b
			b_coefs = self.simi.get_b_freq(10,100,parameters['EL'],parameters['Vreset'],parameters['Vt'],parameters['gL'],parameters['a'],parameters['tauw'])

			for h, hicann in enumerate(hicann_index):

				hicann_array_mean = []
				hicann_array_err = []

				for n, neuron in enumerate(neuron_index[h]):

					# Calcultate b and error
					hicann_array_mean.append(self.poly_process(sorted_array_mean[h][n],b_coefs))
					hicann_array_err.append(self.poly_process(sorted_array_err[h][n],b_coefs))

				processed_array_mean.append(hicann_array_mean)
				processed_array_err.append(hicann_array_err)


		if (parameter == 'Vexp'):
			# Calculate relation between frequency and gL
			Vexp_coefs = self.simi.get_vexp_freq(800,1000,parameters['EL'],parameters['Vreset'],parameters['Vt'],parameters['gL'],parameters['dT'])

			for h, hicann in enumerate(hicann_index):

				hicann_array_mean = []
				hicann_array_err = []

				for n, neuron in enumerate(neuron_index[h]):

					# Calcultate Vexp and error
					hicann_array_mean.append(self.poly_process(sorted_array_mean[h][n],Vexp_coefs))
					hicann_array_err.append(self.poly_process(sorted_array_err[h][n],Vexp_coefs))

				processed_array_mean.append(hicann_array_mean)
				processed_array_err.append(hicann_array_err)
		return processed_array_mean, processed_array_err

	def meassure(self, hicann, neurons, parameter, parameters, value, calibrated_parameters):
		print "Configuring the hardware ..."
		config_start = time.time()
		# If no debug mode, configure the hardware. If debug, do nothing
		if not self.debug_mode:
			self.hwi.configure(hicann, neurons, calibrated_parameters)
		print "Hardware configuration completed in {:.2}s".format(time.time() - config_start)
		#sys.exit()

		# Measure all neurons
		print "Measuring the hardware ..."
		meas_start = time.time()

		# If no debug mode, measure the hardware. If debug, return 0
		if (self.debug_mode == False):
			measurement = self.hwi.measure(hicann, neurons, parameter, parameters, 0, value)
		else:
			measurement = neuron_index
		print "Measurement completed in " + str(time.time() - meas_start) + " s"
		return measurement


	## Process an array with a polynomial function
	# @param array The array of data to be processed
	# @param coefficients The polynomial coefficients
	def poly_process (self,array,coefficients):
		poly_array = []
		for i in array:
			poly_array.append(float(np.polyval(coefficients,i)))
		return poly_array

	def get_steps(self, p):
		# Create input_array
		start, stop = self.parameter_ranges[p]['min'], self.parameter_ranges[p]['max']
		pts = self.parameter_ranges[p]['pts']
		return np.linspace(start, stop, pts)

	## Compute calibration function
	# @param measuredValues The measured values
	# @param values The input values
	def compute_calib_function(self,measuredValues,values):
		calibCoeff = np.polyfit(measuredValues,values,2)
		a = float(calibCoeff[0])
		b = float(calibCoeff[1])
		c = float(calibCoeff[2])
		fit = (a,b,c)
		return fit
