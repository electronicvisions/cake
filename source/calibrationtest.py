## @package calibrationtest
# Calibration tests for the BrainScaleS hardware

# Imports
import calibrationcontroller as cali
import scaledhw as scali
import time
import numpy as np
import pylab

## Class CalibrationTest
class CalibrationTest:

	## The constructor
	# @param hardware Hardware system to use
	# @param fpga Logical number of the FPGA board to use
	# @param scope_address IP address of the scope
	# @param hicann_index Index of the HICANNs to test. Example [1,2]
	# @param neuron_index Index of the neurons to test. Example [[1,2,3],[1,2,3,4]]
	# @param reps Number of repetitions to do
	# @param calibrated Use calibration ?
	# @param run Run the experiment again ?
	# @param save_results Save results ?
	# @param compare Compare calibrated and non calibrated ?
	# @param bio Get results in the biological domain ?
	def __init__(self,hardware,fpga,scope_address,hicann_index,neuron_index,reps,calibrated,run,save_results,compare,bio):

		# Test parameters
		self.nb_reps = reps
		
		# Run ?
		self.run = run
		
		# FPGA number ?
		self.fpga = fpga
		
		# Save results ?
		self.save_results = save_results
		
		# Convert to bio values ?
		self.bio = bio
		
		# Compare results ?
		self.compare = compare
		
		# Neuron index
		self.neuron_index = neuron_index
		
		# Hicann index
		self.hicann_index = hicann_index

		# Default values
		self.parameters = {"EL" : 600.0,	
			"gL" : 1000.0,
			"Vt" : 580.0,
			"Vreset" : 500.0,
			"C" : 2.6,
			"a" : 0.0,
			"tw" : 30.0,
			"b" : 0.0,
			"dT" : 10.0,
			"Vexp" : 1000.0, 
			"expAct" : 0.0,
			"gsynx" : 0.0,
			"gsyni" : 0.0,
			"tausynx" : 10.0,
			"Esynx" : 0.0,
			"tausyni" : 10.0,
			"Esyni" : 0.0,
			"tauref" : 0.}
			
		self.Vt_default_max = 1700 # mV
		self.EL_default_max = 1500 # mV
		self.gL_default = 300 # nA
		self.EL_default = 500 # mV
		self.a_default = 800 # nA
		self.tauw_default = 2000 # nA
		
		self.EL_IF_value = 800 # mV
		self.Vreset_IF_value = 500 # mV
		self.Vt_IF_value = 700 # mV
		self.gL_IF_value = 1000 # nA
		
		self.dT_default = 10 # mV
		self.Vexp_default = 400 # mV
		
		self.Esynx_default = 1000 # mV
		
		self.current_default = 200 # nA
		
		# Calibrated or non-calibrated case
		self.use_calib = calibrated
		
		# Create interfaces
		self.cali = cali.CalibrationController(hardware,scope_address,fpga=fpga)
		self.scali = scali.scaledHW()
	
	## Main test function
	# @param parameter The parameter to test. Example : "EL", "gL"
	def test(self,parameter):

		# Define measurement arrays
		meas_array = []

		meas_array_mean = []
		meas_array_err = []
		calibrated_parameters = []
		
		# Debug arrays
		meas_array_Vreset = []
		meas_array_Vt = []		
		
		# Parameter specific initialization
		if (parameter == 'EL'):
		
			# Put Vt to maximum value to avoid spiking & set gL
			self.parameters["Vt"] = self.Vt_default_max
			self.parameters["gL"] = self.gL_default

			# Value to be tested
			self.parameters["EL"] = 650 # mV
			
		if (parameter == 'Vreset' or parameter == 'Vt'):

			# Set gL & set EL to high value
			self.parameters["EL"] = self.EL_default_max
			self.parameters["gL"] = self.gL_default
			
		if (parameter == 'gL'):
				
			# Set Vreset
			self.parameters["Vreset"] = self.Vreset_IF_value
	
			# Set EL,Vt
			self.parameters["EL"] = self.EL_IF_value
			self.parameters["Vt"] = self.Vt_IF_value

			# Value to be tested
			self.parameters["gL"] = 1000 # nA
			
		if (parameter == 'tauref'):
				
			# Set I&F parameters
			self.parameters["Vreset"] = self.Vreset_IF_value
			self.parameters["EL"] = self.EL_IF_value
			self.parameters["Vt"] = self.Vt_IF_value
			self.parameters["gL"] = self.gL_IF_value
			
			# Value to be tested
			self.parameters["tauref"] = 1 # us
			
		if (parameter == 'a'):
				
			# Set I&F parameters
			self.parameters["EL"] = self.EL_IF_value
			self.parameters["Vreset"] = self.Vreset_IF_value
			self.parameters["Vt"] = self.Vt_IF_value
			self.parameters["gL"] = self.gL_IF_value
			
			# Set tauw to low value
			self.parameters["tauw"] = self.tauw_default
			
			# Value to test
			self.parameters["a"] = 400
			
		if (parameter == 'tw'):
				
			# Set I&F parameters
			self.parameters["EL"] = self.EL_IF_value
			self.parameters["Vreset"] = self.Vreset_IF_value
			self.parameters["Vt"] = self.Vt_IF_value
			self.parameters["gL"] = self.gL_IF_value
			
			# Set a
			self.parameters["a"] = self.a_default
			
		if (parameter == 'b'):
				
			# Set I&F parameters
			self.parameters["EL"] = self.EL_IF_value
			self.parameters["Vreset"] = self.Vreset_IF_value
			self.parameters["Vt"] = self.Vt_IF_value
			self.parameters["gL"] = self.gL_IF_value
			
			# Set a & tauw
			self.parameters["a"] = self.a_default
			self.parameters["tauw"] = self.tauw_default
			
		if (parameter == 'tausynx' or parameter=='tausyni'):
				
			# Set I&F parameters
			self.parameters["EL"] = self.EL_default
			self.parameters["Vreset"] = self.Vreset_IF_value
			self.parameters["Vt"] = self.Vt_default_max
			self.parameters["gL"] = self.gL_IF_value
			
			# Set Esyn
			self.parameters["Esynx"] = self.Esynx_default

		if (parameter == 'dT'):
				
			# Set I&F parameters
			self.parameters["EL"] = self.EL_IF_value
			self.parameters["Vreset"] = self.Vreset_IF_value
			self.parameters["Vt"] = self.Vt_IF_value
			self.parameters["gL"] = self.gL_IF_value
			
			# Set Exp terms
			self.parameters["expAct"] = 2000
			self.parameters["Vexp"] = self.Vexp_default
			
		if (parameter == 'Vexp'):
				
			# Set I&F parameters
			self.parameters["EL"] = self.EL_IF_value
			self.parameters["Vreset"] = self.Vreset_IF_value
			self.parameters["Vt"] = self.Vt_IF_value
			self.parameters["gL"] = self.gL_IF_value
			
			# Set Exp terms
			self.parameters["expAct"] = 2000
			self.parameters["dT"] = self.dT_default
			self.parameters["Vexp"] = self.Vexp_default
			
		if (self.run == True):

			# Get calibrated_parameters
			for h,hicann in enumerate(self.hicann_index):
				
				#Use calibration database
				if (self.use_calib == True):
					calibrated_parameters_hicann = self.scali.scaled_to_hw(hicann,self.neuron_index[h],self.parameters,parameters=parameter+'_test')
				else:
					calibrated_parameters_hicann = self.scali.scaled_to_hw(hicann,self.neuron_index[h],self.parameters,parameter+'_calibration')
						
				calibrated_parameters.append(calibrated_parameters_hicann)
				
			#print calibrated_parameters
			
			for j in range(self.nb_reps):
				
				# Log
				print "Starting repetition number", j+1 
			
				# Configure the system
				start_config = time.time()
				self.cali.hwi.configure(self.hicann_index,self.neuron_index,calibrated_parameters)
				print "Hardware configured in", time.time()-start_config, 's'
				
				# Measure all neurons
				meas_array.append(self.cali.hwi.measure(self.hicann_index,self.neuron_index,parameter,self.parameters))
			
			if (self.save_results == True):		
				np.save('parallel_test_'+ parameter + '_' + str(self.use_calib) + '.npy',meas_array)
		
		if (self.save_results == True):	
			meas_array = np.load('parallel_test_'+ parameter + '_' + str(self.use_calib) + '.npy')
					
		# Calculate mean array
		for h,hicann in enumerate(self.hicann_index):
			
			temp_array_hicann = []		
			
			for n, neuron in enumerate(self.neuron_index[h]):
				temp_array_neuron = []
				
				for k in range(self.nb_reps):
					temp_array_neuron.append(meas_array[k][h][n])
			
				temp_array_hicann.append(float(np.mean(temp_array_neuron)))
			
			meas_array_mean.append(temp_array_hicann)

		# Calculate err array
		for h,hicann in enumerate(self.hicann_index):
			temp_array_hicann = []		
			
			for n, neuron in enumerate(self.neuron_index[h]):
				temp_array_neuron = []
				
				for k in range(self.nb_reps):
					temp_array_neuron.append(meas_array[k][h][n])
			
				temp_array_hicann.append(float(np.std(temp_array_neuron)))
			
			meas_array_err.append(temp_array_hicann)
		
		if (parameter == 'gL' or parameter == 'a' or parameter == 'tauref' or parameter == 'b' or parameter == 'Vexp'):
			print self.parameters
			expected_result = self.cali.simi.compute_freq(30e-6,0,self.parameters)
			if (self.bio == True):
				expected_result = expected_result/1e4
			print 'Expected value :', expected_result
			
		# Plot
		nb_bins = 20
		for h,hicann in enumerate(self.hicann_index):
			
			# Print results
			np.save('neuron_index_' + parameter + '_' + str(self.use_calib) + '.npy', self.neuron_index[h])
			np.save('neuron_mean_' + parameter + '_' + str(self.use_calib) + '.npy', meas_array_mean[h])
		
			if (parameter == 'EL' or parameter == 'Vt' or parameter == 'Vreset'):
				pylab.ylim(0,1800)
				if (self.bio == True):
					for i,item in enumerate(meas_array_mean[h]):
						meas_array_mean[h][i] = meas_array_mean[h][i]/10 -130
						meas_array_err[h][i] = meas_array_err[h][i]/10
						
					pylab.ylim(-100,0)
					
				pylab.errorbar(self.neuron_index[h],meas_array_mean[h],yerr=meas_array_err[h],fmt='ro',c='k')
				pylab.plot(self.neuron_index[h],int(np.mean(meas_array_mean[h]))*np.ones(len(self.neuron_index[h])))
				pylab.xlabel('Neuron number')
				
				pylab.ylabel('Membrane potential [mV]')
				
				pylab.figure()
				pylab.hist(meas_array_err[h],bins=nb_bins)
				pylab.xlabel('Trial-to-trial standard deviation [mV]')
				pylab.ylabel('Count')
				
				pylab.figure()
				pylab.hist(meas_array_mean[h],bins=nb_bins)
				pylab.xlabel('Membrane potential [mV]')
				if (self.bio == True):
					pylab.xlim(-100,0)
				pylab.ylabel('Count')
				
			if (parameter == 'gL' or parameter == 'a' or parameter == 'tauref' or parameter == 'b' or parameter == 'Vexp'):
				if (self.bio == True):
					for i,item in enumerate(meas_array_mean[h]):
						meas_array_mean[h][i] = meas_array_mean[h][i]/1e4
						meas_array_err[h][i] = meas_array_err[h][i]/1e4
						
					#pylab.ylim(0,200)
						
				pylab.errorbar(self.neuron_index[h],meas_array_mean[h],yerr=meas_array_err[h],fmt='ro',c='k')
				pylab.plot(self.neuron_index[h],int(np.mean(meas_array_mean[h]))*np.ones(len(self.neuron_index[h])))
				pylab.xlabel('Neuron number')
				pylab.plot(self.neuron_index[h],expected_result*np.ones(len(self.neuron_index[h])))
				pylab.ylabel('Frequency [Hz]')
				
				pylab.figure()
				pylab.hist(meas_array_err[h],bins=nb_bins)
				pylab.xlabel('Trial-to-trial standard deviation [Hz]')
				pylab.ylabel('Count')
				
				pylab.figure()
				pylab.hist(meas_array_mean[h],bins=nb_bins)
				pylab.xlabel('Frequency [Hz]')
				if (self.bio == True):
					pass
					#pylab.xlim(0,200)
				pylab.ylabel('Count')
				
				
			# printt results
			print 'Mean value over all neurons :', np.mean(meas_array_mean)
			print ''
			print 'Mean value of individual std :', np.mean(meas_array_err)
			print ''
			print 'Standard deviation of means :', np.std(meas_array_mean)
			print ''
			print 'Standard deviation :', np.std(meas_array_mean)/np.mean(meas_array_mean)*100,'%'
			
			#print 'Neuron index', self.neuron_index
			#print ''
			#print 'Mean value for each neuron :', meas_array_mean
			#print ''
			#print 'Error for each neuron :', meas_array_err
			#print ''
				
			# Comparison hist ?
			if (self.compare == True):
				
				# Load other results
				if (self.use_calib == True):
					meas_array = np.load('parallel_test_'+ parameter + '_' + str(False) + '.npy')
				if (self.use_calib == False):
					meas_array = np.load('parallel_test_'+ parameter + '_' + str(True) + '.npy')
				
				# Calculate mean array
				meas_array_mean_second = []
				meas_array_err_second = []
				for h,hicann in enumerate(hicann_index):
					temp_array_hicann = []		
					
					for n, neuron in enumerate(self.neuron_index[h]):
						temp_array_neuron = []
						
						for k in range(self.nb_reps):
							temp_array_neuron.append(meas_array[k][h][n])
					
						temp_array_hicann.append(float(np.mean(temp_array_neuron)))
					
					meas_array_mean_second.append(temp_array_hicann)

				# Calculate err array
				for h,hicann in enumerate(hicann_index):
					temp_array_hicann = []		
					
					for n, neuron in enumerate(self.neuron_index[h]):
						temp_array_neuron = []
						
						for k in range(self.nb_reps):
							temp_array_neuron.append(meas_array[k][h][n])
					
						temp_array_hicann.append(float(np.std(temp_array_neuron)))
					
					meas_array_mean_second.append(temp_array_hicann)
					
				if (parameter == 'EL' or parameter == 'Vt' or parameter == 'Vreset'):
					if (self.bio == True):
						for i,item in enumerate(meas_array_mean_second[h]):
							meas_array_mean_second[h][i] = meas_array_mean_second[h][i]/10 -130
							meas_array_err[h][i] = meas_array_err[h][i]/10
							
				if (parameter == 'gL' or parameter == 'a' or parameter == 'tauref'):
					if (self.bio == True):
						for i,item in enumerate(meas_array_mean_second[h]):
							meas_array_mean_second[h][i] = meas_array_mean_second[h][i]/1e4
							meas_array_mean_second[h][i] = meas_array_mean_second[h][i]/1e4
				
				# Trash weird values if calibrated
				#mean = np.mean(meas_array_mean_second[h])
				#std = np.std(meas_array_mean_second[h])
				#for i,item in enumerate(meas_array_mean_second[h]):
					#if (abs(mean - item) > std):
						#meas_array_mean_second[h].pop(i)
				print np.mean(meas_array_mean_second[h])
				print np.std(meas_array_mean_second[h])
				
				pylab.hist(meas_array_mean_second[h],bins=nb_bins)
				
				# Figure Old vs New
				pylab.figure()
				pylab.scatter(meas_array_mean[h],meas_array_mean_second[h])
				
		pylab.show()
