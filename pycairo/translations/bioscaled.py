## @package bioscaled
# Translation from biological model parameters to scaled parameters with the hardware time & voltage ranges

# Imports
import numpy
import math

## Translation from biological model parameters to scaled parameters with the hardware time & voltage ranges
class bioScaled:

    ## Convert biological parameters to accelerated parameters
    # @param params Dictionnary of neuron parameters, according to pyNN conventions
    # @param autoscale Automatically calculate the voltage scaling and shift factors
    # @param tfactor Acceleration factor
    # @param scfactor Voltage scaling factor
    # @param shfactor Voltage shift factor
    def bio_to_scaled(self,params,tFactor=10000,scFactor=5,shFactor=1200,autoscale=False):
      
        # Compute scFactor and shFactor if autoscale is True
        if (autoscale == True):
            
            # Define HW range and target Vreset
            HW_range = 200 # mV
            target = 500 # mV
            
            # Calculate scFactor
            if (params['v_thresh'] > params['v_spike']):
                max_voltage = params['v_spike']
            else:
                max_voltage = params['v_thresh']
            bio_range = max_voltage - params['v_reset']
            scFactor = HW_range/bio_range
            
            # Calculate shFactor
            shFactor = target - params['v_reset'] * scFactor
            
        # Scaling voltages
        ELS = params['v_rest'] * scFactor + shFactor
        EsynxS = params['e_rev_E'] * scFactor + shFactor
        EsyniS = params['e_rev_I'] * scFactor + shFactor
        VtS = params['v_spike'] * scFactor + shFactor
        VresetS = params['v_reset'] * scFactor + shFactor
        VexpS = params['v_thresh'] * scFactor + shFactor
        dTS = params['delta_T'] * scFactor

        # Scaling membrane time constant
        tauS =  params['tau_m']/tFactor*1000 # in micro-seconds
      
        # Scaling synaptic time constants
        tauSynSx = params['tau_syn_E']/tFactor*1000 # in us
        tauSynSi = params['tau_syn_I']/tFactor*1000 # in us
         
        # Choose correct capacitance
        if (tFactor == 100000):
            CS = 400e-3 # pF
        else:
            CS = 2.6 # pF

        # Calculate new gL
        gLS = CS/tauS*1000 # in micro siemens

        # Scaling adaptation terms
        gL = params['cm']/params['tau_m']*1000 # in micro siemens
        aS = params['a'] * gLS/gL # in nano siemens (as in PyNN)
        twS = params['tau_w']/tFactor*1e3 # in micro seconds
        if (aS == 0):
            bS = 0
        else:
            bS = params['b'] * scFactor * aS/params['a'] # in nano Ampere
       
        # Scaling refractory period
        taurefS = params['tau_refrac']/tFactor*1e3 # in micro seconds
        
        # Scaling synaptic weights
        if (params.has_key('g_syn_E')):
            gsynSx = params['g_syn_E'] * gLS/gL
        else:
            gsynSx = 2000
        if (params.has_key('g_syn_I')):
            gsynSi = params['g_syn_I'] * gLS/gL
        else:
            gsynSi = 2000

        return {
        'gL'     : gLS,     # in uS
        'Esynx'  : EsynxS , # in mV
        'Esyni'  : EsyniS,  # in mV
        'EL'     : ELS,     # in mV
        'Vt'     : VtS,     # in mV
        'Vreset' : VresetS, # in mV 
        'Vexp'   : VexpS,   # in mV
        'dT'     : dTS,     # in mV
        'C'      : CS,      # in pF
        'a'      : aS,      # in nS
        'tw'     : twS,      # in us (micro seconds)
        'b'      : bS,       # in nA
        'tauref' : taurefS,  # in us (micro seconds) 
        'tausynx': tauSynSx, # in us (micro seconds)
        'tausyni': tauSynSi, # in us (micro seconds)
        'expAct' : 1,        # activate exponential term (scalar: 0 or 1) ???
        'gsynx': gsynSx,          # ???
        'gsyni': gsynSi
        }

   # def scaledToBiol(self,params,tFactor,scFactor,shFactor):
# 
#       # List of neuron parameters 
#       ELS = params['v_restS']    
#       gLS = params['g_leakS']
#       VtS = params['v_spikeS']
#       VresetS = params['v_resetS']
#       CmS = params['cmS']
#       aS = params['aS']
#       twS = params['tau_wS']
#       taurefS = params['tau_refracS']
#       bS = params['bS']
#       dTS = params['delta_TS']
#       VexpS = params['v_threshS']
# 
#       # Scaling voltages
#       EL = (ELS - shFactor)/scFactor
#       Vt = (VtS - shFactor)/scFactor
#       Vreset = (VresetS - shFactor)/scFactor
#       Vexp = (VexpS - shFactor)/scFactor
#       dT = dTS/scFactor
# 
#       # Scaling membrane time constant
# 
#       taumS = CmS/gLS
#       taum = taumS*tFactor*1000
# 
#       # Calculating biol capacitance in nF
# 
#       C = 0.2 # nF
# 
#       # Calculating gL
# 
#       gL = C/taum*1000 # nF
# 
#       # Scaling adaptation terms
#       a = aS * gL/gLS    
# 
#       tw = twS*tFactor
# 
#       b = bS * a/aS / scFactor
# 
#       # Scaling ref period
#       tauref = taurefS * tFactor
# 
#       print 'gL', gL
# 
#       return {'v_rest': EL, 'v_spike': Vt, 'v_reset': Vreset, 'v_thresh': Vexp, 'delta_T': dT, 'tau_m': taum, 'a': a, 'tau_w': tw, 'b': b, 'tau_refrac': tauref}
#       
