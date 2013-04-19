## @package adex_neuron
# Documentation for adex_neuron

import numpy
import math

class Neuron:
    '''This class defines the simulation of the AdEx neuron model.'''
    ## Definition of the parameters and variables to be recorded
    def __init__(self):
        self.parameters = {"EL" : 700.0,    
            "gL" : 1000.0,
            "Vt" : 1000.0,
            "Vreset" : 700.0,
            "C" : 2.6,
            "a" : 0.0,
            "tw" : 30.0,
            "b" : 0.0,
            "dT" : 2.0,
            "Vexp" : -50.0, 
            "expAct" : 0.0,
            "gsynx" : 0.0,
            "gsyni" : 0.0,
            "tausynx" : 10.0,
            "tausyni" : 10.0,
            "Esynx" : 1000.0,
            "Esyni" : 200.0,
            "tauref" : 0.0}
            
        self.v = self.parameters['EL']*1e-3
        self.w = 0
        self.gsynx = 0
        self.gsyni = 0
        self.spikes = []
        
        self.record = True
        self.v_record = []
        self.time_count = 0
        self.refrac_count = 0
        self.gsynx_record = []
        self.gsyni_record = []

    ## Main function for the simulation of the AdEx neuron
    # @param timestep The simulation timesteps in seconds
    # @param stim The neuron current stimulus in A
    # @param spikes_x The list of excitatory spikes
    # @param spikes_i The list of excitatory spikes
    # @param ramp_current The value of the ramp current in A
    def sim(self,timestep,stim,spikes_x=[],spikes_i=[],ramp_current=0):
                            
            # Incoming spikes (exc)
            if (int(self.time_count/timestep) in spikes_x):
                self.gsynx = self.gsynx + self.parameters['gsynx']*1e-9
            self.gsynx = self.gsynx - self.gsynx * timestep / (self.parameters['tausynx']*1e-6)

            # Incoming spikes (inh)
            if (int(self.time_count/timestep) in spikes_i):
                self.gsyni = self.gsyni + self.parameters['gsyni']*1e-9
            self.gsyni = self.gsyni - self.gsyni * timestep / (self.parameters['tausyni']*1e-6)
            
            # Refractory period ?
            if (self.refrac_count < 0):
                self.refrac_count = 0
            
            if (self.refrac_count == 0):
                self.v = self.v + (-self.gsynx*(self.v-self.parameters['Esynx']*1e-3) -self.gsyni*(self.v-self.parameters['Esyni']*1e-3) - self.parameters['gL']*1e-9 * (self.v-self.parameters['EL']*1e-3) + self.parameters['expAct']*self.parameters['gL']*1e-9*self.parameters['dT']*1e-3*math.exp((self.v-self.parameters['Vexp']*1e-3)/(self.parameters['dT']*1e-3)) + stim + ramp_current - self.w)*timestep/(self.parameters['C']*1e-12)
                self.w = self.w + (self.parameters['a']*1e-9*(self.v-self.parameters['EL']*1e-3) - self.w)/(self.parameters['tw']*1e-6)*timestep
            
            if (self.refrac_count > 0):
                self.v = self.v
                self.w = self.w

            if (self.v > self.parameters['Vt']*1e-3):
                self.v = self.parameters['Vreset']*1e-3
                self.w = self.w + self.parameters['b']*1e-9
                self.spikes.append(self.time_count)
                self.refrac_count = self.parameters['tauref']*1e-6
               
            self.refrac_count = self.refrac_count - timestep
               
            if (self.record == True):
                self.v_record.append(self.v)
                self.w_record.append(self.w)
                self.gsynx_record.append(self.gsynx)
                self.gsyni_record.append(self.gsyni)

            if (self.time_count == 0):
                self.refrac_count = 0

            # Increment time_count
            self.time_count = self.time_count + timestep
        
    ## Reset the simulation        
    def reset(self):
        
        self.v = self.parameters['EL']*1e-3
        self.w = 0
        self.gsynx = 0
        self.gsyni = 0
        self.spikes = []
        
        self.v_record = []    
        self.w_record = []
        self.time_count = 0
        self.gsynx_record = []
        self.gsyni_record = []
