'''Simulation of the AdEx neuron model by simple integration over a given timestep.'''

import math
import numpy as np
import pylab
import sys

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
    
class Simulator:

    def __init__(self):
        
        # Create neuron
        self.soma = Neuron()
        
        # Parameters variation
        self.parameters_variation = False    
        self.variation = 1e-3
        
        # Noise
        self.noise = False
        self.noise_std = 10e-3

        # Spike time jitter
        self.jitter = False
        self.jitter_std = 1e-8

        # Current ramp value
        self.current_ramp = 0

    def sim(self,time,stim,params,spikes_x=[],spikes_i=[],timestep=1e-8,ramp_current=0):

        # Create time array
        time_array = np.arange(0,time,timestep)

        # Apply parameters variations
        if (self.parameters_variation == True):
            for i in params:
                params[i] = np.random.normal(params[i],self.variation*params[i])
        
        # Apply parameters
        self.soma.parameters = params
        
        # Reset neuron
        self.soma.reset()
        
        # Store stimulus
        applied_stim = stim
        
        # Simulate
        for i in time_array:

            # Generate current ramp
            self.current_ramp = ramp_current*i/time
    
            # Simulation step
            self.soma.sim(timestep,applied_stim,spikes_x,spikes_i,self.current_ramp)
            
        # Add noise
        if (self.noise == True):
            for i,item in enumerate(self.soma.v_record):
                self.soma.v_record[i] = self.soma.v_record[i] + np.random.normal(0,self.noise_std)

        # Add jitter 
        if (self.jitter == True):
            for i,item in enumerate(self.soma.spikes):
                self.soma.spikes[i] = np.random.normal(self.soma.spikes[i],self.jitter_std)

        self.current_ramp = 0
        
        return time_array, self.soma.v_record, self.soma.spikes
    
    # Plot a simulation   
    def plot_sim(self,time,stim,params,spikes_x=[],spikes_i=[]):
       
        t,v,spikes = self.sim(time,stim,params,spikes_x,spikes_i)
        pylab.plot(t,v,linewidth = 1.0)

        #pylab.show()
        
    # Plot spikes only
    def plot_spikes(self,time,stim,params):
       
        t,v,spikes = self.sim(time,stim,params)
        for i in spikes:
            pylab.plot([i,i],[0,1],c='k')
        
        pylab.ylim(-5,6)
        #pylab.show()
   
    # Return only spike times    
    def get_spike_time(self,time,stim,params):
        t,v,spikes = self.sim(time,stim,params)
        return spikes
   
    # Return an array with spiking frequency   
    def compute_freq(self,time,stim,params):
        t,v,spikes = self.sim(time,stim,params)
   
        r = len(spikes)
        for i in range(1,r):
            p = spikes[i]-spikes[i-1]
        try:
            p
            freq = 1/p
        except:
            freq = 0
          
        return freq
        
    # Return an array with spiking frequency   
    def compute_psp_integral(self,time,stim,params):
        
        t,v,spikes = self.sim(time,stim,params,[100])
        
        for j,item in enumerate(v):
            v[j] = v[j]*1e3
        integral = np.trapz(t,v)
          
        return integral
   
    # Plot the spiking frequency versus time   
    def plot_freq(self,time,stim,params):
        t,v,spikes = self.sim(time,stim,params)
        
        freqs = []
        for i in range(1,len(spikes)):
            freqs.append(1/(spikes[i]-spikes[i-1]))
            
        pylab.plot(freqs)
        #pylab.show()
   
    # Return time and frequency arrays   
    def get_time_freq(self,time,stim,params):
        t,f = self.compute_freq(time,stim,params)
        last = len(f) - 1
        f.append(f[last])
        return t,f
    
    # Return the stationnary spiking frequency   
    def get_stat_freq(self,time,stim,params):
        eps = 0.1
        limit = 0
        limitfound = False
        
        t,v,spikes = self.sim(time,stim,params)
        freqs = []
        for i in range(1,len(spikes)):
            freqs.append(1/(spikes[i]-spikes[i-1]))
            
        for i in range(1,len(freqs)):
            diff = freqs[i]-freqs[i-1]
            if (diff < eps):
                limit = freqs[i]
                limitfound = True
        if (limitfound == False):
            print "No limit found !"
        return limit
   
    # Return the discrete derivative   
    def compute_deriv(self,array):
        deriv = range(len(array))
        for i in range(len(array)-1):
            deriv[i] = array[i+1] - array[i]
        return deriv
        

    # Return the relation between gL and frequency
    def get_gl_freq(self,gLMin,gLMax,EL,Vreset,Vt):
        
        # Number of points
        s = 3
        
        # Frequency array
        freq = []

        # Generate gL values
        values = np.arange(gLMin,gLMax+(gLMax-gLMin)/s,(gLMax-gLMin)/s)
      
        # Calculate stationnary frequency for each gL  
        for i in values:
            parameters = {"EL" : EL,    
            "gL" : i,
            "Vt" : Vt,
            "Vreset" : Vreset,
            "C" : 2.6,
            "a" : 1e-6,
            "tw" : 30.0,
            "b" : 1e-6,
            "dT" : 10.0,
            "Vexp" : 1000.0, 
            "expAct" : 1e-6,
            "gsynx" : 1e-6,
            "gsyni" : 1e-6,
            "tausynx" : 10.0,
            "tausyni" : 10.0,
            "tauref" : 0.0,
            "Esynx" : 1e-6,
            "Esyni" : 1e-6}
            freq.append(self.compute_freq(30e-6,0,parameters))
            self.soma.reset()
        
        # Calculate fit
        a,b,c = np.polyfit(freq,values,2)
        
        # For debug : plot fit
        # valuesC = []
        # for i in freq:
            # valuesC.append(a*i*i+b*i+c)
            
        # pylab.scatter(freq,values)
        # pylab.plot(freq,valuesC)
        # pylab.show()
        
        return a,b,c
        
    # # Return the relation between gL and frequency
    # def getTausynPSP(self,paramMin,paramMax,EL,Vreset,Vt,gL,Esynx):
        
    #     # Number of points
    #     s = 5
        
    #     # PSP array
    #     psp = []

    #     # Generate values
    #     values = np.arange(paramMin,paramMax+(paramMax-paramMin)/s,(paramMax-paramMin)/s)
      
    #     for i in values:
    #         parameters = {"EL" : EL,    
    #         "gL" : gL,
    #         "Vt" : Vt,
    #         "Vreset" : Vreset,
    #         "C" : 2.6,
    #         "a" : 1e-6,
    #         "tw" : 30.0,
    #         "b" : 1e-6,
    #         "dT" : 10.0,
    #         "Vexp" : 1000.0, 
    #         "expAct" : 1e-6,
    #         "gsynx" : 100.0,
    #         "tauref" : 0.0,
    #         "tausynx" : i,
    #         "Esynx" : Esynx}
    #         psp.append(self.computePSPIntegral(100e-6,0,parameters))
    #         self.soma.reset()
        
    #     # Calculate fit
    #     a,b,c = np.polyfit(psp,values,2)
        
    #     # For debug : plot fit
    #     # valuesC = []
    #     # for i in psp:
    #         # valuesC.append(a*i*i+b*i+c)
            
    #     # pylab.scatter(psp,values)
    #     # pylab.plot(psp,valuesC)
    #     # pylab.show()
        
    #     return a,b,c
        
    # Return the relation between a and frequency
    def get_a_freq(self,paramMin,paramMax,EL,Vreset,Vt,gL):
        
        # Number of points
        s = 3
        
        # Frequency array
        freq = []

        # Generate values
        values = np.arange(paramMin,paramMax+(paramMax-paramMin)/s,(paramMax-paramMin)/s)
      
        # Calculate stationnary frequency
        for i in values:
            parameters = {"EL" : EL,    
            "gL" : gL,
            "Vt" : Vt,
            "Vreset" : Vreset,
            "C" : 2.6,
            "a" : i,
            "tw" : 1.0,
            "b" : 1e-6,
            "dT" : 300.0,
            "Vexp" : 1000.0, 
            "expAct" : 1e-6,
            "gsynx" : 1e-6,
            "gsyni" : 1e-6,
            "tausynx" : 10.0,
            "tausyni" : 10.0,
            "tauref" : 0.0,
            "Esynx" : 1e-6,
            "Esyni" : 1e-6}
            freq.append(self.get_stat_freq(200e-6,0,parameters))
            self.soma.reset()
        
        # Calculate fit
        a,b,c = np.polyfit(freq,values,2)
        
        # For debug : plot fit
        # valuesC = []
        # for i in freq:
            # valuesC.append(a*i*i+b*i+c)
            
        # pylab.scatter(freq,values)
        # pylab.plot(freq,valuesC)
        # pylab.show()
        
        return a,b,c
        
    # Return the relation between a and frequency
    def get_vexp_freq(self,paramMin,paramMax,EL,Vreset,Vt,gL,dT):
        
        # Number of points
        s = 20
        
        # Frequency array
        freq = []

        # Generate values
        values = np.arange(paramMin,paramMax+(paramMax-paramMin)/s,(paramMax-paramMin)/s)
      
        # Calculate stationnary frequency
        for i in values:
            parameters = {"EL" : EL,    
            "gL" : gL,
            "Vt" : Vt,
            "Vreset" : Vreset,
            "C" : 2.6,
            "a" : 1e-6,
            "tw" : 1.0,
            "b" : 1e-6,
            "dT" : dT,
            "Vexp" : i, 
            "expAct" : 1,
            "gsynx" : 1e-6,
            "gsyni" : 1e-6,
            "tausynx" : 10.0,
            "tausyni" : 10.0,
            "tauref" : 0.0,
            "Esynx" : 1e-6,
            "Esyni" : 1e-6}
            freq.append(self.get_stat_freq(200e-6,0,parameters))
            self.soma.reset()
            
        # Cut constant part
        thresh = 10e3
        new_freq = []
        new_values = []
        for i in range(1,len(freq)):
            if (math.fabs(freq[i] - freq[i-1]) > thresh):
                new_freq.append(freq[i])
                new_values.append(values[i])
                
        freq = new_freq
        values = new_values
                
        # Cut constant part
        thresh = 1e6
        new_freq = []
        new_values = []
        for i in range(len(freq)):
            if (freq[i] < thresh):
                new_freq.append(freq[i])
                new_values.append(values[i])
                
        # print freq
        # print new_freq
        # print new_values
        
        # Calculate fit
        a,b,c = np.polyfit(new_freq,new_values,2)
        
        # For debug : plot fit
        valuesC = []
        for i in new_freq:
            valuesC.append(a*i*i+b*i+c)
            
        pylab.scatter(new_freq,new_values)
        pylab.plot(new_freq,valuesC)
        pylab.show()
        
        return a,b,c
        
    # Return the relation between a and frequency
    def get_b_freq(self,paramMin,paramMax,EL,Vreset,Vt,gL,a,tauw):
        
        # Number of points
        s = 3
        
        # Frequency array
        freq = []

        # Generate gL values
        values = np.arange(paramMin,paramMax+(paramMax-paramMin)/s,(paramMax-paramMin)/s)
      
        # Calculate stationnary frequency for each gL  
        for i in values:
            parameters = {"EL" : EL,    
            "gL" : gL,
            "Vt" : Vt,
            "Vreset" : Vreset,
            "C" : 2.6,
            "a" : a,
            "tw" : tauw,
            "b" : i,
            "dT" : 300.0,
            "Vexp" : 1000.0, 
            "expAct" : 1e-6,
            "tauref" : 0.0,
            "gsynx" : 1e-6,
            "gsyni" : 1e-6,
            "tausynx" : 10.0,
            "tausyni" : 10.0,
            "tauref" : 0.0,
            "Esynx" : 1e-6,
            "Esyni" : 1e-6}
            freq.append(self.get_stat_freq(100e-6,0,parameters))
            self.soma.reset()
        
        # Calculate fit
        a,b,c = np.polyfit(freq,values,2)
        
        # For debug : plot fit
        # valuesC = []
        # for i in freq:
            # valuesC.append(a*i*i+b*i+c)
            
        # pylab.scatter(freq,values)
        # pylab.plot(freq,valuesC)
        # pylab.show()
        
        return a,b,c
        
    # Return the relation between gL and frequency
    def get_tauw_isi(self,paramMin,paramMax,EL,Vreset,Vt,gL,gLadapt,stim):
        
        # Number of points
        s = 10
        
        # Frequency array
        ISI = []

        # Generate gL values
        values = np.arange(paramMin,paramMax+(paramMax-paramMin)/s,(paramMax-paramMin)/s)
      
        # Calculate stationnary frequency for each gL  
        for i in values:
            parameters = {"EL" : EL,    
            "gL" : gL,
            "Vt" : Vt,
            "Vreset" : Vreset,
            "C" : 2.6,
            "a" : gLadapt,
            "tw" : i,
            "b" : 1e-6,
            "dT" : 10.0,
            "Vexp" : 1000.0, 
            "expAct" : 1e-6,
            "gsynx" : 1e-6,    
            "tausynx" : 10.0,
            "tauref" : 0.0,
            "Esynx" : 1e-6}
            t,v,spikes = self.sim(10e-6,stim,parameters)
            freqs = []
            for i in range(1,len(spikes)):
                freqs.append(1/(spikes[i]-spikes[i-1]))
            a,b = np.polyfit(range(len(freqs)),freqs,1)
            ISI.append(b)
            #ISI.append(spikes[1]-spikes[0])
            #print parameters['tw']
            #print spikes
            self.soma.reset()
            
        # for i in range(len(values)):
            # values[i] = float(1/float(values[i]))
        
        # Calculate fit
        a,b,c,d,e = np.polyfit(ISI,values,4)
            
        #For debug : plot fit
        # valuesC = []
        # for i in ISI:
            # valuesC.append(a*i*i*i*i+b*i*i*i+c*i*i+d*i+e)
            
        # pylab.scatter(ISI,values)
        # pylab.plot(ISI,valuesC)
        # pylab.show()
        
        return a,b,c,d,e
        
        # Return the relation between gL and frequency
    def get_tauw_int(self,paramMin,paramMax,EL,Vreset,Vt,gL,gLadapt,stim):
        
        # Number of points
        s = 10
        
        # Integral array
        Int = []

        # Generate gL values
        values = np.arange(paramMin,paramMax+(paramMax-paramMin)/s,(paramMax-paramMin)/s)
      
        # Calculate stationnary frequency for each gL  
        for i in values:
            parameters = {"EL" : EL,    
            "gL" : gL,
            "Vt" : Vt,
            "Vreset" : Vreset,
            "C" : 2.6,
            "a" : gLadapt,
            "tw" : i,
            "b" : 1e-6,
            "dT" : 10.0,
            "Vexp" : 1000.0, 
            "expAct" : 1e-6,
            "gsynx" : 1e-6,    
            "tausynx" : 10.0,
            "tauref" : 0.0,
            "Esynx" : 1e-6}
            t,v,spikes = self.sim(50e-6,stim,parameters)
            Int.append(np.trapz(t,v))
            self.soma.reset()
        
        # Calculate fit
        a,b,c = np.polyfit(Int,values,2)
            
        #For debug : plot fit
        # valuesC = []
        # for i in ISI:
            # valuesC.append(a*i*i*i*i+b*i*i*i+c*i*i+d*i+e)
            
        # pylab.scatter(ISI,values)
        # pylab.plot(ISI,valuesC)
        # pylab.show()
        
        return a,b,c

    def smooth(self,v,width):

        s_v = []
        for i in range(width-1):
            s_v.append(v[i])
        for i in range(width-1,len(v)):
            s = 0
            for j in range(width):
                s = s + v[i-j]
            s_v.append(s/width)
        return s_v

