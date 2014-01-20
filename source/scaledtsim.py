## @package scaledtsim
# Documentation for scaledTsim

# Imports
import numpy
import pylab
import math

# Class scaledTsim
class scaledTsim:

	## The constructor
	# @param process The process used in the simulations. Options are "180" or "65"
	def __init__ (self,process='180'):
		self.process = process

	## Convert scaled AdEx parameters to transistor level simulation parameters
	# @param params The neuron parameters to convert
	def convert(self,params):
	
		HW_parameters = {'EL': 1000,
		'Vt': 1000,
		'Vexp': 536,
		'Vreset': 500,
		'Esynx': 900,
		'Esyni': 900,
		'Vsynx': 0,
		'Vsyni': 0,
		'tausynx': 900,
		'tausyni': 900,
		'Iconvi': 0,
		'expAct': 0,
		'b': 0,
		'a': 0,
		'gL': 400,
		'tauref': 2000,
		'tw': 2000,
		"gsynx" : 0.0,
		"gsyni" : 0.0,
		'Iintbbx': 0,
		'Iintbbi': 0,
		'dT': 750,
		'Ispikeamp': 2000
		}
		if (self.process == '180'):
			# Don't change
			HW_parameters['Vreset'] = params['Vreset']
			
			HW_parameters['Esyni'] = params['Esyni']
			HW_parameters['Esynx'] = params['Esynx']
			
			# From transistor-level simulations
			HW_parameters['EL'] = 1.02*params['EL'] - 8.58
			HW_parameters['Vt'] = 0.998*params['Vt'] - 3.55
			HW_parameters['Esynx'] = 1.02*params['Esynx'] - 8.58
			HW_parameters['gL'] = 5.52e-5*params['gL']*params['gL'] + 0.24*params['gL'] + 0.89
			HW_parameters['tauref'] = 1/(0.025*params['tauref'] - 0.0004)
		
			HW_parameters['a'] = 4.93e-5*params['a']*params['a'] + 0.26*params['a'] - 0.66
			if (params['a'] ==0):
				HW_parameters['a'] = 0
			HW_parameters['b'] = -0.14*params['b']*params['b'] + 45*params['b'] + 54.75
			if (params['b'] ==0):
				HW_parameters['b'] = 0
			HW_parameters['tw'] = 1/(-4.4e-6*params['tw']*params['tw'] + 0.00032*params['tw'] - 0.0005)
			
			HW_parameters['dT'] = 9.2385890861*params['dT']*params['dT'] + 66.3846854343*params['dT'] - 94.2540733183
			HW_parameters['Vexp'] = 0.371990571979*params['Vexp'] + 100.290157129 # for dT = 8 mV
			if (params['Vexp'] > HW_parameters['Vt']):
				HW_parameters['Vexp'] = params['Vexp']
			
			#HW_parameters['expAct'] = params['expAct']
			#HW_parameters['gsynx'] = params['gsynx']
			HW_parameters['tausynx'] = -3.94*params['tausynx']*params['tausynx'] + 37*params['tausynx'] + 1382
			HW_parameters['tausyni'] = -3.94*params['tausyni']*params['tausyni'] + 37*params['tausyni'] + 1382
			
		if (self.process == '65'):
			# Don't change
			HW_parameters['Vreset'] = params['Vreset']
			
			HW_parameters['Esyni'] = params['Esyni']
			HW_parameters['Esynx'] = params['Esynx']
			
			# From transistor-level simulations
			HW_parameters['EL'] = 1.006*params['EL'] - 3.76
			HW_parameters['Vt'] = 1.06*params['Vt'] - 30.47
			HW_parameters['Esynx'] = 1.02*params['Esynx'] - 8.58
			HW_parameters['gL'] = 2.98e-5*params['gL']*params['gL'] + 0.126*params['gL'] + 9.48
			HW_parameters['tauref'] = 1/(0.025*params['tauref'] - 0.0004)
			
			HW_parameters['a'] = 2.5978e-5*params['a']*params['a'] + 0.1446*params['a'] + 11.9216318971
			if (params['a'] == 0):
				HW_parameters['a'] = 0
			HW_parameters['b'] = -0.14*params['b']*params['b'] + 45*params['b'] + 54.75
			if (params['b'] == 0):
				HW_parameters['b'] = 0
			HW_parameters['tw'] = 1/(-2.90677723331e-05*params['tw']*params['tw'] +0.00394594735745*params['tw'] - 0.00280076949178)
			
			HW_parameters['dT'] = 9.2385890861*params['dT']*params['dT'] + 66.3846854343*params['dT'] - 94.2540733183
			HW_parameters['Vexp'] = 0.371990571979*params['Vexp'] + 100.290157129 # for dT = 8 mV
			if (params['Vexp'] > HW_parameters['Vt']):
				HW_parameters['Vexp'] = params['Vexp']
			
			#HW_parameters['expAct'] = params['expAct']
			#HW_parameters['gsynx'] = params['gsynx']
			HW_parameters['tausynx'] = -3.94*params['tausynx']*params['tausynx'] + 37*params['tausynx'] + 1382
			HW_parameters['tausyni'] = -3.94*params['tausyni']*params['tausyni'] + 37*params['tausyni'] + 1382

		return HW_parameters
