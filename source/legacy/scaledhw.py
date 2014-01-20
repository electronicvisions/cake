## @package scaledhw
# Translation from scaled AdEx parameters to hardware parameters

# Imports
import numpy
from numpy import polyval
import pylab
import math
import databaseinterface as DB


## Main class
class scaledHW:

	ident = [1,0]
	# polynom coefficients of the ideal ADEX transformation.
	# copied from branch "ideal" of method scaled_to_hw(...)
	IDEAL_TRAFO_ADEX = {
        # Don't change
        'Vreset' : ident,
        'EL' : ident,
        'Esyni' : ident,
        'Esynx' : ident,
        # From transistor-level simulations
        'EL'      : [1.02, -8.58], # FIXME: WHY TWO TIMES EL?
        'Vt'      : [0.998, -3.55],
        'Esynx'   : [1.02, -8.58], # FIXME: WHY TWO TIMES Esynx?
        'gL'      : [5.52e-5, 0.24, 0.89],
        'tauref'  : [0.025, -0.0004], # INVERSE
        'a'       : [4.93e-5, 0.26, -0.66],
        'b'       : [-0.14, 45, 54.75],
        'tw'      : [-4.4e-6, 0.00032, -0.0005], # INVERSE
        'dT'      : [136, -131.6],
        'Vexp'    : [0.64, 93.15],
        'tausynx' : [-3.94, 37, 1382],
        'tausyni' : [-3.94, 37, 1382],
        }
	# default values for technical ADEX params
    # (taken from 'ideal' branch of scaled_to_hw(..))
	DEFAULT_ADEX = {
		'expAct'  : 2000,
		'gsynx'   : 1000,
		'gsyni'   : 1000,
		'Iintbbx' : 1500,
		'Iintbbi' : 1500,
		}
	# polynom coefficients of the ideal LIF transformation.
	# copied from branch "ideal_LIF" of method scaled_to_hw(...)
	IDEAL_TRAFO_LIF = {
        # Don't change
        'Vreset' : ident,
        'EL' : ident,
        'Esyni' : ident,
        'Esynx' : ident,
        # From transistor-level simulations
        'EL'      : [1.02, -8.58], # FIXME: WHY TWO TIMES EL?
        'Vt'      : [0.998, -3.55],
        'Esynx'   : [1.02, -8.58], # FIXME: WHY TWO TIMES Esynx?
        'gL'      : [5.52e-5, 0.24, 0.89],
        'tauref'  : [0.025, -0.0004], # INVERSE
        'tausynx' : [-3.94, 37, 1382],
        'tausyni' : [-3.94, 37, 1382],
        }
	# default values for LIF config
    # technical params and disabled adex params
    # (taken from 'ideal_LIF' branch of scaled_to_hw(..))
	DEFAULT_LIF = {
        'a'       : 0,
        'b'       : 0,
        'tw'      : 1000,
        'dT'      : 1000,
        'Vexp'    : 1200,
        'expAct'  : 0,
        'gsynx'   : 2000,
        'gsyni'   : 2000,
        'Iintbbx' : 2000,
        'Iintbbi' : 2000,
        }

	# The constructor
	# @param use_db Use the database ?
	# @param use_fallback Use the ideal transformation for single parameters for single not-yet-calibrated parameters?
	def __init__ (self, use_db=True, use_fallback=False):

		# Use calibration database ?
		if (use_db==True): 
			self.dbi = DB.DatabaseInterface()
			#print "Connected to calibration database"
		self.use_fallback = use_fallback
		
	## Convert one parameter for a given neuron, from the scaled domain to the hardware domain
	# @param neuron The desired neuron 
	# @param param The parameter to convert
	# @param value The value of the parameter to convert
	def scaled_to_hw_single(self,neuron,param,value): 

		# If the parameter is not marked as calibrated and option use_fallback is set,
		# use the ideal adex transformation
		if not neuron[param + "_calibrated"] and self.use_fallback:
			return polyval(self.IDEAL_TRAFO_ADEX[param],value)

		fit = neuron[param + "_fit"]
		
		if (fit != []):  
			return fit[0]*value*value + fit[1]*value + fit[2]
		else:
			return value

	## Convert one parameter for a given neuron, from the hardware domain to the scaled domain
	# @param neuron The desired neuron 
	# @param param The parameter to convert
	# @param value The value of the parameter to convert
	def hw_to_scaled_single(self,neuron,param,value):
		
		fit = neuron[param + "_fit"]
		
		if (fit != []):      

			a = fit[0]
			b = fit[1]
			c = fit[2]
			
			# Calculate reverse function
			delta = b*b - 4*a*(c-value)
			
			if (delta >= 0):
			
				return (-b + numpy.sqrt(delta))/(2*a)
				
			else:
				return value
		
		else:
			return value

    ## Convert scaled AdEx parameters to HW parameters for one HICANN
    # @param hicann Number of the desired HICANN
    # @param neuron_index Neuron index. Example : [[1,2,3],[1,2,3]]
    # @param params The parameters to convert
    # @param parameters The conversion option. Use "ideal" to not use calibration.
	def scaled_to_hw(self,hicann,neuron_index,params,parameters='all'):
		
		# Get all neurons for HICANN hicann
		if (parameters == 'direct' or parameters == 'ideal' or parameters == 'ideal_LIF'):
				pass
		else:
			neurons = self.dbi.get_neurons(hicann)
		
		# List of HW parameters
		hicann_parameters = []
		
		for n in neuron_index:
	
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
		
			# Get neuron data if required
			#print params
			if (parameters == 'direct' or parameters == 'ideal' or parameters == 'ideal_LIF'):
				pass
			else:
				currentNeuron = neurons[n]
				
			if (parameters=='EL_calibration'):
				HW_parameters['EL'] = params['EL']
				HW_parameters['Vt'] = params['Vt']
				HW_parameters['gL'] = params['gL']
				
			if (parameters=='EL_test'):
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'])
				HW_parameters['Vt'] = 1700
				HW_parameters['gL'] = params['gL']

			if (parameters=='Vt_calibration'):
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'])
				HW_parameters['Vt'] = params['Vt']
				HW_parameters['Vreset'] = params['Vt'] - 200
				
			if (parameters=='Vt_test'):
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'])
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'])
				HW_parameters['Vreset'] = params['Vt'] - 200
				
			if (parameters=='Vreset_calibration'):
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'])
				HW_parameters['Vreset'] = params['Vreset']
				HW_parameters['Vt'] = params['Vreset'] + 200
				
			if (parameters=='Vreset_test'):
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'])
				HW_parameters['Vreset'] = self.scaled_to_hw_single(currentNeuron,"Vreset",params['Vreset'])
				HW_parameters['Vt'] = params['Vreset'] + 200
				
			if (parameters=='gL_calibration'):
				HW_parameters['gL'] = params['gL']
				HW_parameters['Vreset'] = params['Vreset']
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				
				#HW_parameters['Vreset'] = params['Vreset']
				#HW_parameters['EL'] = params['EL']
				#HW_parameters['Vt'] = params['Vt']
				
			if (parameters=='gL_test'):
				HW_parameters['Vreset'] = params['Vreset']
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['gL'] = self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				
				#HW_parameters['Vreset'] = params['Vreset']
				#HW_parameters['EL'] = params['EL']
				#HW_parameters['Vt'] = params['Vt']
				
			if (parameters=='dT_calibration'):
				HW_parameters['dT'] = params['dT']
				HW_parameters['Vexp'] = params['Vexp']
				HW_parameters['gL'] = self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['Vreset'] = 500
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',500)
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				HW_parameters["expAct"] = 2000
				
			if (parameters=='dT_test'):
				HW_parameters['dT'] = self.scaled_to_hw_single(currentNeuron,"dT",params['dT'])
				HW_parameters['Vexp'] = params['Vexp']
				HW_parameters['gL'] =  self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['Vreset'] = 500
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',500)
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				HW_parameters["expAct"] = 2000
				
			if (parameters=='Vexp_calibration'):
				HW_parameters['Vreset'] = 500
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',500)
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['dT'] = self.scaled_to_hw_single(currentNeuron,"dT",params['dT'])
				HW_parameters['Vexp'] = params['Vexp'] - corr_factor
				HW_parameters['gL'] =  self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				HW_parameters["expAct"] = 2000
				
			if (parameters=='Vexp_test'):
				HW_parameters['Vreset'] = 500
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',500)
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['dT'] = self.scaled_to_hw_single(currentNeuron,"dT",params['dT'])
				HW_parameters['Vexp'] = self.scaled_to_hw_single(currentNeuron,"Vexp",params['Vexp'] - corr_factor)
				HW_parameters['gL'] =  self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				HW_parameters["expAct"] = 2000
				
			if (parameters=='tausynx_calibration'):
				HW_parameters['gL'] = self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['Vreset'] = 500
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'])
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'])
				
				HW_parameters['Esynx'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esynx'])
				HW_parameters['tausynx'] = params['tausynx']
				HW_parameters['gsynx'] = 1000
				HW_parameters['gsyni'] = 0
				HW_parameters['Iintbbx'] = 1500
				
			if (parameters=='tausynx_test'):
				HW_parameters['gL'] = self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['Vreset'] = 500
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'])
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'])
				
				HW_parameters['Esynx'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esynx'])
				HW_parameters['tausynx'] = params['tausynx']
				HW_parameters['gsynx'] = 1000
				HW_parameters['gsyni'] = 0
				HW_parameters['Iintbbx'] = 1000

				HW_parameters['tausynx'] = self.scaled_to_hw_single(currentNeuron,"tausynx",params['tausynx'])

			if (parameters=='tausyni_calibration'):
				HW_parameters['gL'] = self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['Vreset'] = 500
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'])
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'])
				
				HW_parameters['Esyni'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esyni'])
				HW_parameters['tausyni'] = params['tausyni']
				HW_parameters['gsyni'] = 1000
				HW_parameters['gsynx'] = 0
				HW_parameters['Iintbbi'] = 1000
				
			if (parameters=='tausyni_test'):
				HW_parameters['gL'] = self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['Vreset'] = 500
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'])
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'])
				
				HW_parameters['Esyni'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esyni'])
				HW_parameters['tausyni'] = params['tausyni']
				HW_parameters['gsyni'] = 1000
				HW_parameters['gsynx'] = 0
				HW_parameters['Iintbbi'] = 1000

				HW_parameters['tausyni'] = self.scaled_to_hw_single(currentNeuron,"tausyni",params['tausyni'])
				
				
			if (parameters=='tauref_calibration'):
				
				HW_parameters['tauref'] = params['tauref']
				
				HW_parameters['Vreset'] = params['Vreset']
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['gL'] = self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				
			if (parameters=='tauref_test'):
				HW_parameters['tauref'] = self.safe_invert_tauref( self.scaled_to_hw_single(currentNeuron,"tauref",params['tauref']) )
				HW_parameters['Vreset'] = params['Vreset']
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['gL'] = self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				
				
			if (parameters=='a_calibration'):
				HW_parameters['gL'] = self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['a'] = params['a']
				HW_parameters['Vreset'] = 500
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',500)
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				
			if (parameters=='a_test'):
				HW_parameters['gL'] = self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['a'] = self.scaled_to_hw_single(currentNeuron,"a",params['a'])
				HW_parameters['Vreset'] = 500
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',500)
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				
			if (parameters=='b_calibration'):
				HW_parameters['gL'] = self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['a'] = self.scaled_to_hw_single(currentNeuron,"a",params['a'])
				HW_parameters['b'] = params['b']
				HW_parameters['tw'] = self.scaled_to_hw_single(currentNeuron,"tw",params['tw'])
				HW_parameters['Vreset'] = 500
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',500)
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				
			if (parameters=='b_test'):
				HW_parameters['gL'] = self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['a'] = self.scaled_to_hw_single(currentNeuron,"a",params['a'])
				HW_parameters['b'] = self.scaled_to_hw_single(currentNeuron,"b",params['b'])
				HW_parameters['tw'] = self.scaled_to_hw_single(currentNeuron,"tw",params['tw'])
				HW_parameters['Vreset'] = 500
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',500)
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				
			if (parameters=='tw_calibration'):
				HW_parameters['gL'] = self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['a'] = self.scaled_to_hw_single(currentNeuron,"a",params['a'])
				HW_parameters['tw'] = params['tw']
				HW_parameters['Vreset'] = 500
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',500)
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				
			if (parameters=='tw_test'):
				HW_parameters['gL'] = self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['a'] = self.scaled_to_hw_single(currentNeuron,"a",params['a'])
				HW_parameters['tw'] = self.scaled_to_hw_single(currentNeuron,"tw",params['tw'])
				HW_parameters['Vreset'] = 500
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',500)
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				
			if (parameters=='None'):
			
				HW_parameters['Vreset'] = 500
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',500)
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['gL'] = params['gL']
				HW_parameters['EL'] = params['EL']
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				
			if (parameters=='direct'):
				
				HW_parameters['Vreset'] = params['Vreset']
				HW_parameters['gL'] = params['gL']
				HW_parameters['EL'] = params['EL']
				HW_parameters['Vt'] = params['Vt']
				
				HW_parameters['a'] = params['a']
				HW_parameters['b'] = params['b']
				HW_parameters['tw'] = params['tw']
				
				HW_parameters['dT'] = params['dT']
				HW_parameters['Vexp'] = params['Vexp']
				HW_parameters['expAct'] = params['expAct']
				
				HW_parameters['gsynx'] = params['gsynx']
				HW_parameters['gsyni'] = params['gsynx']
				
			if (parameters=='ideal'):
				
				# Don't change
				HW_parameters['Vreset'] = params['Vreset']
				HW_parameters['EL'] = params['EL']
				
				HW_parameters['Esyni'] = params['Esyni']
				HW_parameters['Esynx'] = params['Esynx']
				
				# From transistor-level simulations
				HW_parameters['EL'] = 1.02*params['EL'] - 8.58
				HW_parameters['Vt'] = 0.998*params['Vt'] - 3.55
				HW_parameters['Esynx'] = 1.02*params['Esynx'] - 8.58
				HW_parameters['gL'] = 5.52e-5*params['gL']*params['gL'] + 0.24*params['gL'] + 0.89
				if (params['tauref'] == 0):
					HW_parameters['tauref'] = 2000
				else:
                    # trafo for tauref is inverted
					inverse_hw_value = (0.025*params['tauref'] - 0.0004)
					HW_parameters['tauref'] = self.safe_invert_tauref(inverse_hw_value)
			
				HW_parameters['a'] = 4.93e-5*params['a']*params['a'] + 0.26*params['a'] - 0.66
				if (params['a'] == 0):
					HW_parameters['a'] = 0
				HW_parameters['b'] = -0.14*params['b']*params['b'] + 45*params['b'] + 54.75
				if (params['b'] ==0):
					HW_parameters['b'] = 0
				HW_parameters['tw'] = 1/(-4.4e-6*params['tw']*params['tw'] + 0.00032*params['tw'] - 0.0005)
				
				HW_parameters['dT'] = 136*params['dT'] - 131.6
				HW_parameters['Vexp'] = 93.15 + 0.64*params['Vexp']
				if (params['Vexp'] > params['Vt']):
					HW_parameters['Vexp'] = HW_parameters['Vt']
				
				HW_parameters['expAct'] = 2000
				HW_parameters['gsynx'] = 1000
				HW_parameters['gsyni'] = 1000
				HW_parameters['tausynx'] = -3.94*params['tausynx']*params['tausynx'] + 37*params['tausynx'] + 1382
				HW_parameters['tausyni'] = -3.94*params['tausyni']*params['tausyni'] + 37*params['tausyni'] + 1382
				HW_parameters['Iintbbx'] = 1500
				HW_parameters['Iintbbi'] = 1500

			if (parameters=='ideal_refactored'):
				# Don't change
				HW_parameters['Vreset'] = params['Vreset']
				HW_parameters['EL'] = params['EL']
				
				HW_parameters['Esyni'] = params['Esyni']
				HW_parameters['Esynx'] = params['Esynx']
				
				# From transistor-level simulations
				HW_parameters['EL'] = polyval(self.IDEAL_TRAFO_ADEX['EL'], params['EL'])
				HW_parameters['Vt'] = polyval(self.IDEAL_TRAFO_ADEX['Vt'], params['Vt'])
				HW_parameters['Esynx'] = polyval(self.IDEAL_TRAFO_ADEX['Esynx'], params['Esynx'])
				HW_parameters['gL'] = polyval(self.IDEAL_TRAFO_ADEX['gL'], params['gL'])
				# SPECIAL HANDLING of some params
				if (params['tauref'] == 0):
					HW_parameters['tauref'] = 2000
				else:
					# trafo for tauref is inverted
					inverse_hw_value =  polyval(self.IDEAL_TRAFO_ADEX['tauref'], params['tauref'])
					HW_parameters['tauref'] = self.safe_invert_tauref(inverse_hw_value)
			
				if (params['a'] == 0):
					HW_parameters['a'] = 0
				else:
					HW_parameters['a'] = polyval(self.IDEAL_TRAFO_ADEX['a'], params['a'])

				if (params['b'] ==0):
					HW_parameters['b'] = 0
				else:
					HW_parameters['b'] = polyval(self.IDEAL_TRAFO_ADEX['b'], params['b'])

				HW_parameters['tw'] = 1./polyval(self.IDEAL_TRAFO_ADEX['tw'], params['tw'])
				
				HW_parameters['dT'] = polyval(self.IDEAL_TRAFO_ADEX['dT'], params['dT'])
				HW_parameters['Vexp'] = polyval(self.IDEAL_TRAFO_ADEX['Vexp'], params['Vexp'])
				if (params['Vexp'] > params['Vt']):
					HW_parameters['Vexp'] = HW_parameters['Vt']

				HW_parameters['tausynx'] = polyval(self.IDEAL_TRAFO_ADEX['tausynx'], params['tausynx'])
				HW_parameters['tausyni'] = polyval(self.IDEAL_TRAFO_ADEX['tausyni'], params['tausyni'])
				
				# TECHNICAL DEFAULTS
				HW_parameters['expAct']  = self.DEFAULT_ADEX['expAct']
				HW_parameters['gsynx']   = self.DEFAULT_ADEX['gsynx']
				HW_parameters['gsyni']   = self.DEFAULT_ADEX['gsyni']
				HW_parameters['Iintbbx'] = self.DEFAULT_ADEX['Iintbbx']
				HW_parameters['Iintbbi'] = self.DEFAULT_ADEX['Iintbbi']


			if (parameters=='ideal_LIF'):
				
				# Don't change
				HW_parameters['Vreset'] = params['Vreset']
				HW_parameters['EL'] = params['EL']
				
				HW_parameters['Esyni'] = params['Esyni']
				HW_parameters['Esynx'] = params['Esynx']
				
				# From transistor-level simulations
				HW_parameters['EL'] = 1.02*params['EL'] - 8.58
				HW_parameters['Vt'] = 0.998*params['Vt'] - 3.55
				HW_parameters['Esynx'] = 1.02*params['Esynx'] - 8.58
				HW_parameters['gL'] = 5.52e-5*params['gL']*params['gL'] + 0.24*params['gL'] + 0.89
				if (params['tauref'] == 0):
					HW_parameters['tauref'] = 2000
				else:
                    # trafo for tauref is inverted
					inverse_hw_value = (0.025*params['tauref'] - 0.0004)
					HW_parameters['tauref'] = self.safe_invert_tauref(inverse_hw_value)
				
				HW_parameters['a'] = 0
				HW_parameters['b'] = 0
				HW_parameters['tw'] = 1000
				
				HW_parameters['dT'] = 1000
				HW_parameters['Vexp'] = 1200
				
				HW_parameters['expAct'] = 0
				HW_parameters['gsynx'] = 2000
				HW_parameters['gsyni'] = 2000
				HW_parameters['Iintbbx'] = 2000
				HW_parameters['Iintbbi'] = 2000


				HW_parameters['tausynx'] = -3.94*params['tausynx']*params['tausynx'] + 37*params['tausynx'] + 1382
				HW_parameters['tausyni'] = -3.94*params['tausyni']*params['tausyni'] + 37*params['tausyni'] + 1382
							
			if (parameters=='ideal_LIF_refactored'):
				
				# Don't change
				HW_parameters['Vreset'] = params['Vreset']
				HW_parameters['EL'] = params['EL']
				
				HW_parameters['Esyni'] = params['Esyni']
				HW_parameters['Esynx'] = params['Esynx']
				
				# From transistor-level simulations
				HW_parameters['EL'] = polyval(self.IDEAL_TRAFO_LIF['EL'], params['EL'])
				HW_parameters['Vt'] = polyval(self.IDEAL_TRAFO_LIF['Vt'], params['Vt'])
				HW_parameters['Esynx'] = polyval(self.IDEAL_TRAFO_LIF['Esynx'], params['Esynx'])
				HW_parameters['gL'] = polyval(self.IDEAL_TRAFO_LIF['gL'], params['gL'])
				# SPECIAL HANDLING of some params
				if (params['tauref'] == 0):
					HW_parameters['tauref'] = 2000
				else:
					# trafo for tauref is inverted
					inverse_hw_value =  polyval(self.IDEAL_TRAFO_LIF['tauref'], params['tauref'])
					HW_parameters['tauref'] = self.safe_invert_tauref(inverse_hw_value)

				HW_parameters['tausynx'] = polyval(self.IDEAL_TRAFO_LIF['tausynx'], params['tausynx'])
				HW_parameters['tausyni'] = polyval(self.IDEAL_TRAFO_LIF['tausyni'], params['tausyni'])
				
				# DEFAULTS and params disabling adaption and exponential term
				HW_parameters['a']  = self.DEFAULT_LIF['a']
				HW_parameters['b']  = self.DEFAULT_LIF['b']
				HW_parameters['tw']  = self.DEFAULT_LIF['tw']

				HW_parameters['dT']  = self.DEFAULT_LIF['dT']
				HW_parameters['Vexp']  = self.DEFAULT_LIF['Vexp']
				
				HW_parameters['expAct']  = self.DEFAULT_LIF['expAct']
				HW_parameters['gsynx']   = self.DEFAULT_LIF['gsynx']
				HW_parameters['gsyni']   = self.DEFAULT_LIF['gsyni']
				HW_parameters['Iintbbx'] = self.DEFAULT_LIF['Iintbbx']
				HW_parameters['Iintbbi'] = self.DEFAULT_LIF['Iintbbi']

							
			if (parameters=='LIF'):
				HW_parameters['Vreset'] = params['Vreset']
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',params['Vreset'])
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				
				HW_parameters['expAct'] = 0
				
				HW_parameters['gL'] = 400
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				
				if (params['tauref'] == 0):
					HW_parameters['tauref'] = 2000
				else:
				    HW_parameters['tauref'] = self.safe_invert_tauref( self.scaled_to_hw_single(currentNeuron,"tauref",params['tauref']) )
				
				HW_parameters['a'] = 0
				HW_parameters['b'] = 0
				
				HW_parameters['Esynx'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esynx'] - corr_factor)
				HW_parameters['tausynx'] = 1400
				HW_parameters['gsynx'] = 2000
				HW_parameters['Iintbbx'] = 1500
				
				HW_parameters['Esyni'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esyni'] - corr_factor)
				HW_parameters['tausyni'] = 1400
				HW_parameters['gsyni'] = 2000
				HW_parameters['Iintbbi'] = 1500
				
			if (parameters=='sampling'):
				
				# Use calibration for voltages
				HW_parameters['Vreset'] = params['Vreset']
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				
				# gL high, tauref low
				HW_parameters['gL'] = 500
				HW_parameters['tauref'] = 10

				HW_parameters['a'] = 0
				HW_parameters['b'] = 0
				HW_parameters['tw'] = 1000
				
				HW_parameters['dT'] = 1600
				HW_parameters['Vexp'] = 1600
				
				HW_parameters['expAct'] = 0
				HW_parameters['Esynx'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esynx'] - corr_factor)
				HW_parameters['gsynx'] = 2000
				HW_parameters['Iintbbx'] = 1500
				
				HW_parameters['Esyni'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esyni'] - corr_factor)
				HW_parameters['gsyni'] = 2000
				HW_parameters['Iintbbi'] = 1500
				
				# Ideal transformation for synaptic time constants
				HW_parameters['tausynx'] = 1500
				HW_parameters['tausyni'] = 1500
				
			if (parameters=='sampling_base'):
				
				# Use calibration for voltages
				HW_parameters['Vreset'] = params['Vreset']
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)
				
				# gL high, tauref low
				HW_parameters['gL'] = 2400
				HW_parameters['tauref'] = 2000

				HW_parameters['a'] = 0
				HW_parameters['b'] = 0
				HW_parameters['tw'] = 1000
				
				HW_parameters['dT'] = 1000
				HW_parameters['Vexp'] = 1200
				
				HW_parameters['expAct'] = 0
				HW_parameters['Esynx'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esynx'] - corr_factor)
				HW_parameters['gsynx'] = 2000
				HW_parameters['Iintbbx'] = 1500
				
				HW_parameters['Esyni'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esyni'] - corr_factor)
				HW_parameters['gsyni'] = 2000
				HW_parameters['Iintbbi'] = 1500
				
				# Ideal transformation for synaptic time constants
				HW_parameters['tausynx'] = -3.94*params['tausynx']*params['tausynx'] + 37*params['tausynx'] + 1382
				HW_parameters['tausyni'] = -3.94*params['tausyni']*params['tausyni'] + 37*params['tausyni'] + 1382
				
			if (parameters=='all'):
				HW_parameters['Vreset'] = 500
				real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',500)
				corr_factor = HW_parameters['Vreset'] - real_Vreset
				
				HW_parameters['dT'] = self.scaled_to_hw_single(currentNeuron,"dT",params['dT'])
				HW_parameters['Vexp'] = self.scaled_to_hw_single(currentNeuron,"Vexp",params['Vexp'] - corr_factor)
				
				HW_parameters['gL'] =  self.scaled_to_hw_single(currentNeuron,"gL",params['gL'])
				HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'] - corr_factor)
				HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)

				HW_parameters['tauref'] = self.safe_invert_tauref( self.scaled_to_hw_single(currentNeuron,"tauref",params['tauref']) )
				
				HW_parameters['a'] = self.scaled_to_hw_single(currentNeuron,"a",params['a'])
				HW_parameters['b'] = self.scaled_to_hw_single(currentNeuron,"b",params['b'])
				HW_parameters['tw'] = self.scaled_to_hw_single(currentNeuron,"tw",params['tw'])
					
				HW_parameters['Esynx'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esynx'] - corr_factor)
				HW_parameters['tausynx'] = self.scaled_to_hw_single(currentNeuron,"tausynx",params['tausynx'])
				HW_parameters['gsynx'] = 2000 # BV: changed from Iconvx to gsynx for consistency
				
			hicann_parameters.append(HW_parameters)
	  
		return hicann_parameters

	## Safely convert the inverted hardware parameter 'tauref' into the required hardware value
    # Returns the inverse of the supplied value if the value is positive otherwise returns 2500., which is
    # the maximum value for FG-current cells. This check assures that very small refractory times dont lead to 
    # a hardware value corresponding to a long refractory time
	# @param inverse_hw_value the inverse hardware value
	def safe_invert_tauref(self, inverse_hw_value):
		# If the result of "scaled_to_hw_single(...)" is negative (which can happen for short requested refactory times,
		# then 1/scaled_to_hw_single(...) is also negative. Hence a value of 0 will be written to into the Floating Gates.
		# but in general is valid: the larger the FG-value, the shorter is the refractory time constant
		# The following check assures a correct transformation
		if inverse_hw_value > 0.:
			return 1./inverse_hw_value
		else:
			return 2500. # the maximum value of the FG-current cells. (TODO: get this from somwhere)
