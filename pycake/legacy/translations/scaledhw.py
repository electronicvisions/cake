'''Translation from scaled AdEx parameters to hardware parameters'''

from numpy import polyval, sqrt
from pycake.interfaces import database as DB
from pycake.config import hardware as dflt
from pycake.config import idealtrafo as ideal


class scaledHW(object):
    def __init__ (self, use_db=True, use_fallback=False):
        '''Args:
            use_db (bool): Use the database?
            use_fallback (bool): Use the ideal transformation for single parameters for single not-yet-calibrated parameters?
        '''

        if use_db:
            self.dbi = DB.DatabaseInterface()
        self.use_fallback = use_fallback

    def scaled_to_hw_single(self,neuron,param,value):
        '''Convert one parameter for a given neuron, from the scaled domain to the hardware domain

        Args:
            neuron: desired neuron
            param: parameter to convert
            value: value of the parameter to convert
    '''

        # If the parameter is not marked as calibrated and option use_fallback is set,
        # use the ideal adex transformation
        if not neuron[param + "_calibrated"] and self.use_fallback:
            return polyval(ideal.IDEAL_TRAFO_ADEX[param],value)

        fit = neuron[param + "_fit"]

        if (fit != []):
            return fit[0]*value*value + fit[1]*value + fit[2]
        else:
            return value

    def hw_to_scaled_single(self,neuron,param,value):
        '''Convert one parameter for a given neuron, from the hardware domain to the scaled domain

        Args:
            neuron: desired neuron
            param: parameter to convert
            value: value of the parameter to convert
        '''

        fit = neuron[param + "_fit"]
        if (fit != []):
            a = fit[0]
            b = fit[1]
            c = fit[2]

            # Calculate reverse function
            delta = b*b - 4*a*(c-value)

            if (delta >= 0):
                return (-b + sqrt(delta))/(2*a)

        return value

    def scaled_to_hw_multi(self,neuron,param_names,params, v_corr_factor=None):
        '''Convert several parameters for a given neuron, from the scaled domain to the hardware domain

        Args:
            neuron: desired neuron
            param_names: a list of parameters to convert
            params: a dictionary, that contains the values for each name in param_names
            v_corr_factor: voltage correction factor that is added to the scaled voltage parameters (['EL', 'Vt', 'Esynx','Esyni','Vexp']) before transformation.

        Returns:
            A dictionary with param_names as key and the hardware parameters as keys.
        '''

        rv = {}
        for key in param_names:
            if v_corr_factor is not None and key in ['EL', 'Vt', 'Esynx','Esyni','Vexp']:
                rv[key] = self.scaled_to_hw_single(neuron,key,params[key] + v_corr_factor)
            else:
                rv[key] = self.scaled_to_hw_single(neuron,key,params[key])
        return rv

    def scaled_to_hw(self,hicann,neuron_index,params,parameters='all'):
        '''Convert scaled AdEx parameters to HW parameters for one HICANN

        Args:
            hicann: Number of the desired HICANN
            neuron_index: Neuron index. Example : [[1,2,3],[1,2,3]]
            params: The parameters to convert
            parameters: The conversion option. Use "ideal" to not use calibration.
        '''

        if (parameters == 'direct' or parameters == 'ideal' or parameters == 'ideal_LIF'):
            pass
        else:
            neurons = self.dbi.get_neurons(hicann) # Get all neurons for HICANN hicann

        hicann_parameters = [] # List of HW parameters

        for n in neuron_index:

            HW_parameters = dflt.get_HW_parameters()

            if parameters in ('direct', 'ideal', 'ideal_LIF'):
                pass
            else:
                currentNeuron = neurons[n] # Get neuron data if required

            if (parameters=='EL_calibration'):
                HW_parameters = dflt.get_HW_parameters()
                HW_parameters['EL'] = params['EL']
                HW_parameters['Vt'] = params['Vt']
                HW_parameters['gL'] = params['gL']

            elif (parameters=='EL_test'):
                HW_parameters = dflt.get_HW_parameters()
                HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'])
                HW_parameters['Vt'] = 1700
                HW_parameters['gL'] = params['gL']

            elif (parameters=='Vt_calibration'):
                HW_parameters = dflt.get_HW_parameters()
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["EL","Vt"],params) )
                HW_parameters['Vreset'] = params['Vt'] - 200

            elif (parameters=='Vreset_calibration'):
                HW_parameters = dflt.get_HW_parameters()
                HW_parameters['EL'] = self.scaled_to_hw_single(currentNeuron,"EL",params['EL'])
                HW_parameters['Vreset'] = params['Vreset']
                HW_parameters['Vt'] = params['Vreset'] + 200

            elif (parameters=='Vreset_test'):
                HW_parameters = dflt.get_HW_parameters()
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["EL","Vreset"],params) )
                HW_parameters['Vt'] = params['Vreset'] + 200

            elif (parameters=='gL_calibration'):
                HW_parameters = dflt.get_HW_parameters()
                HW_parameters['gL'] = params['gL']
                HW_parameters['Vreset'] = params['Vreset']
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset

                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["EL","Vt"],params, v_corr_factor= -corr_factor) )

            elif (parameters=='gL_test'):
                HW_parameters = dflt.get_HW_parameters()
                HW_parameters['Vreset'] = params['Vreset']
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL", "EL","Vt"],params, v_corr_factor= -corr_factor) )

            elif (parameters=='dT_calibration'):
                HW_parameters = dflt.get_HW_parameters(exp=True)
                HW_parameters['dT'] = params['dT']
                HW_parameters['Vexp'] = params['Vexp']
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL", "EL","Vt"],params, v_corr_factor= -corr_factor) )

            elif (parameters=='dT_test'):
                HW_parameters = dflt.get_HW_parameters(exp=True)
                HW_parameters['Vexp'] = params['Vexp']
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt","dT"],params, v_corr_factor= -corr_factor) )

            elif (parameters=='Vexp_calibration'):
                HW_parameters = dflt.get_HW_parameters(exp=True)
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters['Vexp'] = params['Vexp'] - corr_factor
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt","dT"],params, v_corr_factor= -corr_factor) )

            elif (parameters=='Vexp_test'):
                HW_parameters = dflt.get_HW_parameters(exp=True)
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt","dT","Vexp"],params, v_corr_factor= -corr_factor) )

            elif (parameters=='tausynx_calibration'):
                HW_parameters = dflt.get_HW_parameters(syn_in_exc=True)
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt"],params) )

                HW_parameters['Esynx'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esynx']) # TODO: Clarify: using "EL" calibration for 'Esynx'
                HW_parameters['tausynx'] = params['tausynx']

                HW_parameters['Iintbbx'] = 1500 # TODO differs from default

            elif (parameters=='tausynx_test'):
                HW_parameters = dflt.get_HW_parameters(syn_in_exc=True)
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt","tausynx"],params) )

                HW_parameters['Esynx'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esynx']) # TODO: Clarify: using "EL" calibration for 'Esynx'
                HW_parameters['Iintbbx'] = 1000 # TODO differs from default AND from the one in calibration

            elif (parameters=='tausyni_calibration'):
                HW_parameters = dflt.get_HW_parameters(syn_in_inh=True)
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt"],params) )

                HW_parameters['Esyni'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esyni']) # TODO: Clarify: using "EL" calibration for 'Esyni'
                HW_parameters['tausyni'] = params['tausyni']
                HW_parameters['Iintbbi'] = 1000 # TODO differs from default

            elif (parameters=='tausyni_test'):
                HW_parameters = dflt.get_HW_parameters(syn_in_inh=True)
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt","tausyni"],params) )

                HW_parameters['Esyni'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esyni']) # TODO: Clarify: using "EL" calibration for 'Esyni'
                HW_parameters['Iintbbi'] = 1000 # TODO differs from default

            elif (parameters=='tauref_calibration'):
                HW_parameters = dflt.get_HW_parameters()

                HW_parameters['tauref'] = params['tauref']

                HW_parameters['Vreset'] = params['Vreset'] # TODO: why do we use an external param
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt"],params, v_corr_factor= -corr_factor) )

            elif (parameters=='tauref_test'):
                HW_parameters = dflt.get_HW_parameters()

                HW_parameters['Vreset'] = params['Vreset'] # TODO: why do we use an external param
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt"],params, v_corr_factor= -corr_factor) )
                HW_parameters['tauref'] = self.safe_invert_tauref( self.scaled_to_hw_single(currentNeuron,"tauref",params['tauref']) )


            elif (parameters=='a_calibration'):
                HW_parameters = dflt.get_HW_parameters()
                HW_parameters['a'] = params['a']
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt"],params, v_corr_factor= -corr_factor) )

            elif (parameters=='a_test'):
                HW_parameters = dflt.get_HW_parameters()
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt","a"],params, v_corr_factor= -corr_factor) )

            elif (parameters=='b_calibration'):
                HW_parameters = dflt.get_HW_parameters()
                HW_parameters['b'] = params['b']
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt","a","tw"],params, v_corr_factor= -corr_factor) )
                # TODO: shouldnt we use an inverse trafo for "tw"?

            elif (parameters=='b_test'):
                HW_parameters = dflt.get_HW_parameters()
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt","a","tw","b"],params, v_corr_factor= -corr_factor) )
                # TODO: shouldnt we use an inverse trafo for "tw"?

            elif (parameters=='tw_calibration'):
                HW_parameters = dflt.get_HW_parameters()
                HW_parameters['tw'] = params['tw']
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt","a"],params, v_corr_factor= -corr_factor) )

            elif (parameters=='tw_test'):
                HW_parameters = dflt.get_HW_parameters()
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt","a","tw"],params, v_corr_factor= -corr_factor) )
                # TODO: shouldnt we use an inverse trafo for "tw"?

            elif (parameters=='None'):
                HW_parameters = dflt.get_HW_parameters()
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters['gL'] = params['gL']
                HW_parameters['EL'] = params['EL']
                HW_parameters['Vt'] = self.scaled_to_hw_single(currentNeuron,"Vt",params['Vt'] - corr_factor)

            elif (parameters=='direct'):
                HW_parameters = dflt.get_HW_parameters()

                HW_parameters['Vreset'] = params['Vreset']
                HW_parameters['gL'] = params['gL']
                HW_parameters['EL'] = params['EL']
                HW_parameters['Vt'] = params['Vt']

                HW_parameters['a'] = params['a']
                HW_parameters['b'] = params['b']
                HW_parameters['tw'] = params['tw']

                HW_parameters['dT'] = params['dT']
                HW_parameters['Vexp'] = params['Vexp']
                HW_parameters['Ibexp'] = params['Ibexp']

                HW_parameters['gsynx'] = params['gsynx']
                HW_parameters['gsyni'] = params['gsynx']

            elif (parameters=='ideal'):
                HW_parameters = dflt.get_HW_parameters(exp=True, syn_in_exc=True, syn_in_inh=True)

                # straight forwared ideal trafo for the most params
                for param in ["gL","EL","Vt", "Vreset", "dT", "Vexp","a","b",'tausynx', 'tausyni', 'Esynx','Esyni']:
                    HW_parameters[param] = polyval(ideal.IDEAL_TRAFO_ADEX[param],params[param])

                # special treatment for tauref
                if (params['tauref'] == 0):
                    HW_parameters['tauref'] = 2000
                else:
                    # trafo for tauref is inverted
                    inverse_hw_value = polyval(ideal.IDEAL_TRAFO_ADEX['tauref'],params['tauref'])
                    HW_parameters['tauref'] = self.safe_invert_tauref(inverse_hw_value)

                # trafo for tw is inverted
                inverse_hw_value_tw = polyval(ideal.IDEAL_TRAFO_ADEX['tw'],params['tw'])
                HW_parameters['tw'] = self.safe_invert_tauref(inverse_hw_value_tw)

                # special handling for a or b being 0
                if (params['a'] == 0):
                    HW_parameters['a'] = 0
                if (params['b'] ==0):
                    HW_parameters['b'] = 0
                # Vexp should not be greater than Vt
                if (params['Vexp'] > params['Vt']):
                    HW_parameters['Vexp'] = HW_parameters['Vt']

                HW_parameters['Iintbbx'] = 1500 # TODO: differs from default
                HW_parameters['Iintbbi'] = 1500 # TODO: differs from default


            elif (parameters=='ideal_LIF'):
                HW_parameters = dflt.get_HW_parameters(exp=False, syn_in_exc=True, syn_in_inh=True)

                # straight forwared ideal trafo for the most params
                for param in ["gL","EL","Vt", "Vreset", 'tausynx', 'tausyni', 'Esynx','Esyni']:
                    HW_parameters[param] = polyval(ideal.IDEAL_TRAFO_ADEX[param],params[param])

                # special treatment for tauref
                if (params['tauref'] == 0):
                    HW_parameters['tauref'] = 2000
                else:
                    # trafo for tauref is inverted
                    inverse_hw_value = polyval(ideal.IDEAL_TRAFO_ADEX['tauref'],params['tauref'])
                    HW_parameters['tauref'] = self.safe_invert_tauref(inverse_hw_value)

                HW_parameters['tw'] = 1000 # TODO: differs from default

                HW_parameters['dT'] = 1000 # TODO: differs from default
                HW_parameters['Vexp'] = 1200 # TODO: differs from default

                HW_parameters['gsynx'] = 2000 # TODO: differs from default
                HW_parameters['gsyni'] = 2000 # TODO: differs from default


            elif (parameters=='LIF'):
                HW_parameters = dflt.get_HW_parameters(exp=False, syn_in_exc=True, syn_in_inh=True)
                HW_parameters['Vreset'] = params['Vreset']
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',params['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset

                HW_parameters['gL'] = 400 # TODO: WTF? a fixed value for gL
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["EL","Vt"],params, v_corr_factor= -corr_factor) )

                if (params['tauref'] == 0):
                    HW_parameters['tauref'] = 2000
                else:
                    HW_parameters['tauref'] = self.safe_invert_tauref( self.scaled_to_hw_single(currentNeuron,"tauref",params['tauref']) )

                HW_parameters['Esynx'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esynx'] - corr_factor)
                HW_parameters['Esyni'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esyni'] - corr_factor)

                HW_parameters['tausynx'] = 1400 # TODO: differs from default
                HW_parameters['gsynx'] = 2000 # TODO: differs from default
                HW_parameters['Iintbbx'] = 1500 # TODO: differs from default

                HW_parameters['tausyni'] = 1400 # TODO: differs from default
                HW_parameters['gsyni'] = 2000 # TODO: differs from default
                HW_parameters['Iintbbi'] = 1500 # TODO: differs from default

            elif (parameters=='sampling'):
                HW_parameters = dflt.get_HW_parameters()

                # Use calibration for voltages
                HW_parameters['Vreset'] = params['Vreset']
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["EL","Vt"],params, v_corr_factor= -corr_factor) )

                # gL high, tauref low
                HW_parameters['gL'] = 500
                HW_parameters['tauref'] = 10

                HW_parameters['a'] = 0
                HW_parameters['b'] = 0
                HW_parameters['tw'] = 1000

                HW_parameters['dT'] = 1600
                HW_parameters['Vexp'] = 1600

                HW_parameters['Ibexp'] = 0
                HW_parameters['Esynx'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esynx'] - corr_factor)
                HW_parameters['gsynx'] = 2000
                HW_parameters['Iintbbx'] = 1500

                HW_parameters['Esyni'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esyni'] - corr_factor)
                HW_parameters['gsyni'] = 2000
                HW_parameters['Iintbbi'] = 1500

                # Ideal transformation for synaptic time constants
                HW_parameters['tausynx'] = 1500
                HW_parameters['tausyni'] = 1500

            elif (parameters=='sampling_base'):
                HW_parameters = dflt.get_HW_parameters()

                # Use calibration for voltages
                HW_parameters['Vreset'] = params['Vreset']
                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset
                HW_parameters.update( self.scaled_to_hw_multi(currentNeuron, ["EL","Vt"],params, v_corr_factor= -corr_factor) )

                # gL high, tauref low
                HW_parameters['gL'] = 2400
                HW_parameters['tauref'] = 2000

                HW_parameters['a'] = 0
                HW_parameters['b'] = 0
                HW_parameters['tw'] = 1000

                HW_parameters['dT'] = 1000
                HW_parameters['Vexp'] = 1200

                HW_parameters['Ibexp'] = 0
                HW_parameters['Esynx'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esynx'] - corr_factor)
                HW_parameters['gsynx'] = 2000
                HW_parameters['Iintbbx'] = 1500

                HW_parameters['Esyni'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esyni'] - corr_factor)
                HW_parameters['gsyni'] = 2000
                HW_parameters['Iintbbi'] = 1500

                # Ideal transformation for synaptic time constants
                HW_parameters['tausynx'] = -3.94*params['tausynx']*params['tausynx'] + 37*params['tausynx'] + 1382
                HW_parameters['tausyni'] = -3.94*params['tausyni']*params['tausyni'] + 37*params['tausyni'] + 1382

            elif (parameters=='all'):
                HW_parameters = dflt.get_HW_parameters(exp=True,syn_in_exc=True,syn_in_inh=True)

                real_Vreset = self.hw_to_scaled_single(currentNeuron,'Vreset',HW_parameters['Vreset'])
                corr_factor = HW_parameters['Vreset'] - real_Vreset

                HW_parameters.update(self.scaled_to_hw_multi(currentNeuron, ["gL","EL","Vt","dT","Vexp","a","b","tw","tausynx"],params, v_corr_factor= -corr_factor))

                HW_parameters['tauref'] = self.safe_invert_tauref(self.scaled_to_hw_single(currentNeuron,"tauref",params['tauref']))

                HW_parameters['Esynx'] = self.scaled_to_hw_single(currentNeuron,"EL",params['Esynx'] - corr_factor)
                HW_parameters['gsynx'] = 2000 # TODO: differs from default
                HW_parameters['Iintbbx'] = 0 # TODO: differs from default
                HW_parameters['Iintbbi'] = 0 # TODO: differs from default
                HW_parameters['Ibexp'] = 0 # TODO: differs from default
                HW_parameters['gsyni'] = 0 # TODO: differs from default

            hicann_parameters.append(HW_parameters)

        return hicann_parameters

    def safe_invert_tauref(self, inverse_hw_value):
        '''Safely convert the inverted hardware parameter 'tauref' into the required hardware value

        Args:
            inverse_hw_value: the inverse hardware value

        Returns:
            The inverse of the supplied value if the value is positive otherwise returns 2500., which is
            the maximum value for FG-current cells. This check assures that very small refractory times dont lead to
            a hardware value corresponding to a long refractory time
        '''

        # If the result of "scaled_to_hw_single(...)" is negative (which can happen for short requested refactory times,
        # then 1/scaled_to_hw_single(...) is also negative. Hence a value of 0 will be written to into the Floating Gates.
        # but in general is valid: the larger the FG-value, the shorter is the refractory time constant
        # The following check assures a correct transformation
        if inverse_hw_value > 0.:
            return 1./inverse_hw_value
        else:
            return 2500. # the maximum value of the FG-current cells. (TODO: get this from somewhere)
