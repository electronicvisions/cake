'''Simulation of the AdEx neuron model by simple integration over a given timestep.'''

import math

class Neuron:
    '''This class defines the simulation of the AdEx neuron model.'''
    def __init__(self):
        # default set of parameters, will usually be overwritten
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
        self.record = True # store v, w, gsynx, gsyni for every timestep?

        self.reset()

    ## Main function for the simulation of the AdEx neuron
    def sim(self,timestep,stim,spikes_x=[],spikes_i=[],ramp_current=0):
        '''Perform a single timestep.

        Args:
            timestep: simulation timestep in seconds
            stim: neuron current stimulus in A
            spikes_x: optional list of excitatory spikes
            spikes_i: optional list of excitatory spikes
            ramp_current: optional ramp current in A
        '''

        # Incoming spikes (excitatory)
        if (int(self.time_count/timestep) in spikes_x):
            self.gsynx = self.gsynx + self.parameters['gsynx']*1e-9
        self.gsynx = self.gsynx - self.gsynx * timestep / (self.parameters['tausynx']*1e-6)

        # Incoming spikes (inhibitory)
        if (int(self.time_count/timestep) in spikes_i):
            self.gsyni = self.gsyni + self.parameters['gsyni']*1e-9
        self.gsyni = self.gsyni - self.gsyni * timestep / (self.parameters['tausyni']*1e-6)

        if self.refrac_count == 0:
            # integrate
            self.v += ( - self.gsynx*(self.v-self.parameters['Esynx']*1e-3)
                        - self.gsyni*(self.v-self.parameters['Esyni']*1e-3)
                        - self.parameters['gL']*1e-9*(self.v-self.parameters['EL']*1e-3)
                        + self.parameters['expAct']*self.parameters['gL']*1e-9*self.parameters['dT']*1e-3*math.exp((self.v-self.parameters['Vexp']*1e-3)/(self.parameters['dT']*1e-3))
                        + stim + ramp_current
                        - self.w
                      ) * timestep/(self.parameters['C']*1e-12)
            self.w += (self.parameters['a']*1e-9*(self.v-self.parameters['EL']*1e-3) - self.w) * timestep/(self.parameters['tw']*1e-6)

        if self.v > self.parameters['Vt']*1e-3:
            # spike threshold reached
            self.v = self.parameters['Vreset']*1e-3
            self.w = self.w + self.parameters['b']*1e-9
            self.spikes.append(self.time_count)
            self.refrac_count = self.parameters['tauref']*1e-6

        self.refrac_count = self.refrac_count - timestep
        if self.refrac_count < 0:
            self.refrac_count = 0

        if self.record: # record simulation variables
            self.v_record.append(self.v)
            self.w_record.append(self.w)
            self.gsynx_record.append(self.gsynx)
            self.gsyni_record.append(self.gsyni)

        self.time_count = self.time_count + timestep

    def reset(self):
        '''Reset the simulation.'''
        # runtime variables
        self.v = self.parameters['EL']*1e-3
        self.w = 0.
        self.gsynx = 0.
        self.gsyni = 0.

        # simulation time and remaining refraction time
        self.time_count = 0.
        self.refrac_count = 0.

        # recorded spikes and simulation variables
        self.spikes = []
        self.v_record = []
        self.w_record = []
        self.gsynx_record = []
        self.gsyni_record = []
