## @package experimentcontroller
# Python interface to start experiment on the hardware

#### Imports ####
import sys
import bioscaled as bioS
import sca;edhw as SHW
import calibrationcontroller as cali
import time
import numpy as np
import pylab

## Class ExperimentController
class ExperimentController:

	def __init__ (self,hardware='demonstrator'):

		# Initialize interfaces
		self.bios = bioS.bioScaled()
		self.shw = SHW.scaledHW()
		self.cali = cali.CalibrationController(hardware)

	def run_experiment(self,hicann_index,neuron_index,parameters,stimulus,duration,conf_option,meas_option,spike_stimulus=[],scaling_parameters=[],configure=True):

		# Convert into scaled parameters
		if (scaling_parameters == []):
			scaled_p = self.bios.bio_to_scaled(parameters,autoscale=True)
		else:
			scaled_p = self.bios.bio_to_scaled(parameters,tFactor=10000,scFactor=scaling_parameters[0],shFactor=scaling_parameters[1],autoscale=False)
		
		# Convert into hardware parameters
		calibrated_parameters = []
			
		for h,hicann in enumerate(hicann_index):
			
			# Calibrated parameters array for one HICANN
			calibrated_parameters_hicann = self.cali.scali.scaled_to_hw(hicann,neuron_index[h],scaled_p,parameters=conf_option)
			
			calibrated_parameters.append(calibrated_parameters_hicann)

		#print calibrated_parameters
		#sys.exit()

		# Configure HW
		if (configure == True):
			print "Configuring HW ..."
			self.cali.hwi.configure(hicann_index,neuron_index,calibrated_parameters)
			print "HW configuration completed"

		# Measure neurons
		print "Starting measurement ..."
		return self.cali.hwi.measure(hicann_index,neuron_index,meas_option,parameters,stimulus,spike_stimulus=spike_stimulus)
		print "Measurement completed"
	
