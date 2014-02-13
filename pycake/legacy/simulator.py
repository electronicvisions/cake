# -*- coding: utf-8 -*-

"""Simulation of the AdEx neuron model by simple integration
over a given timestep."""

import math
import numpy as np

import pycake.config.simulator as config


class Neuron(object):
    """This class defines the simulation of the AdEx neuron model."""

    def __init__(self):
        self.parameters = {}  # overwritten by Simulator.run_simulation()
        self.record = True  # store v, w, gsynx, gsyni for every timestep?

    def sim_step(self, timestep, stim, spikes_x=[], spikes_i=[], ramp_current=0):
        """Perform a single timestep.

        Args:
            timestep: simulation timestep in seconds
            stim: neuron current stimulus in A
            spikes_x: optional list of excitatory spikes
            spikes_i: optional list of excitatory spikes
            ramp_current: optional ramp current in A
        """

        p = self.parameters

        # Incoming spikes (excitatory)
        if (int(self.time_count / timestep) in spikes_x):
            self.gsynx = self.gsynx + p['gsynx'] * 1e-9
        self.gsynx -= self.gsynx * timestep / (p['tausynx'] * 1e-6)

        # Incoming spikes (inhibitory)
        if (int(self.time_count / timestep) in spikes_i):
            self.gsyni = self.gsyni + p['gsyni'] * 1e-9
        self.gsyni -= self.gsyni * timestep / (p['tausyni'] * 1e-6)

        if self.refrac_count == 0:
            # integrate
            self.v += (- self.gsynx * (self.v - p['Esynx'] * 1e-3)
                       - self.gsyni * (self.v - p['Esyni'] * 1e-3)
                       - p['gL'] * 1e-9 * (self.v - p['EL'] * 1e-3)
                       + p['gL'] * 1e-9 * p['dT'] * 1e-3 * math.exp((self.v - p['Vexp'] * 1e-3) / (p['dT'] * 1e-3))
                       + stim + ramp_current
                       - self.w
                       ) * timestep / (p['C'] * 1e-12)
            self.w += (p['a'] * 1e-9 * (self.v - p['EL'] * 1e-3) - self.w) * timestep / (p['tw'] * 1e-6)

        if self.v > p['Vt'] * 1e-3:
            # spike threshold reached
            self.v = p['Vreset'] * 1e-3
            self.w = self.w + p['b'] * 1e-9
            self.spikes.append(self.time_count)
            self.refrac_count = p['tauref'] * 1e-6

        self.refrac_count = self.refrac_count - timestep
        if self.refrac_count < 0:
            self.refrac_count = 0

        if self.record:  # record simulation variables
            self.v_record.append(self.v)
            self.w_record.append(self.w)
            self.gsynx_record.append(self.gsynx)
            self.gsyni_record.append(self.gsyni)

        self.time_count = self.time_count + timestep

    def reset(self):
        """Reset the simulation parameters and delete recorded data."""

        # runtime variables
        self.v = self.parameters['EL'] * 1e-3  # convert mV to V
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


class Simulator(object):
    """Performs simulations for given parameters to get frequencies etc."""

    def __init__(self):
        """Create Neuron object for simulations."""

        self.neuron = Neuron()

    def run_simulation(self, time, stim, params,
                       spikes_x=[], spikes_i=[],
                       timestep=1e-8, ramp_current=0):
        """Perform a single simulation run for given parameters.

        Args:
            time: end time of simulation run
            stim: neuron current stimulus in A
            params (dict): parameters and values that differ from Neuron's default parameters
            spikes_x (optional list): times of excitatory spikes
            spikes_i (optional list): times of inhibitory spikes
            timestep (optional): simulation timestep
            ramp_current (optional): maximum current, ramped from 0 to this value over full simulation time

        Returns:
            (time, voltage, spikes)

            time: list of simulation timesteps
            voltage: list of voltages at each timestep
            spikes: list of detected spikes
        """

        time_array = np.arange(0, time, timestep)  # array containing each timestep

        if config.parameters_variation:
            # Apply parameters variations
            for i in params:
                params[i] = np.random.normal(params[i], config.variation * params[i])

        self.neuron.parameters = params  # apply parameters

        self.neuron.reset()

        applied_stim = stim  # Store stimulus

        # Simulate
        for i in time_array:
            current_ramp = ramp_current * i / time  # Generate current ramp

            # Simulation step
            self.neuron.sim_step(timestep, applied_stim, spikes_x, spikes_i, current_ramp)

        if config.noise:  # add noise
            for i, item in enumerate(self.neuron.v_record):
                self.neuron.v_record[i] = self.neuron.v_record[i] + np.random.normal(0, config.noise_std)

        if config.jitter:  # add jitter
            for i, item in enumerate(self.neuron.spikes):
                self.neuron.spikes[i] = np.random.normal(self.neuron.spikes[i], config.jitter_std)

        self.current_ramp = 0

        return time_array, self.neuron.v_record, self.neuron.spikes

    def compute_freq(self, time, stim, params):
        """Run a simulation and return the spiking frequency.

        Deprecated, but still used in pycake.Controller
        """

        t, v, spikes = self.run_simulation(time, stim, params)

        freqs = self.compute_frequencies(spikes)
        return freqs[0]

    def compute_frequencies(self, spikes):
        """Compute frequencies between each two spikes in an array.

        Args:
            spikes: array of spike times

        Returns:
            array of frequencies
        """

        freqs = []
        for i in range(1, len(spikes)):
            dt = spikes[i] - spikes[i - 1]
            if dt != 0:
                freqs.append(1 / dt)
            else:
                freqs.append(0)

        return freqs

    def compute_psp_integral(self, time, stim, params):
        """Return an array with spiking frequency"""

        t, v, spikes = self.run_simulation(time, stim, params, [100])

        for j, item in enumerate(v):
            v[j] = v[j] * 1e3
        integral = np.trapz(t, v)

        return integral

    def compute_deriv(self, array):
        """Return the discrete derivative"""

        deriv = range(len(array))
        for i in range(len(array) - 1):
            deriv[i] = array[i + 1] - array[i]
        return deriv

    def get_stat_freq(self, time, stim, params, epsilon=0.1):
        """Get the stationary spiking frequency.

        Runs a simulation using the given parameters.
        Finds the first spiking frequency which has a difference
        smaller than epsilon to its previous frequency.

        Returns:
            stationary spiking frequency
        """

        t, v, spikes = self.run_simulation(time, stim, params)
        freqs = self.compute_frequencies(spikes)

        for i in range(1, len(freqs)):
            if freqs[i] - freqs[i - 1] < epsilon:
                return freqs[i]

        print "No limit found!"
        return 0

    def parameter_range(self, pMin, pMax, num_steps):
        """Generate parameter value range with equal steps between pMin and pMax.

        Last value in range might be greater than pMax.

        Args:
            pMin: minimum value
            pMax: maximum value
            num_steps: number of values in range

        Returns:
            numpy array ranging from pMin to pMax
        """

        stepsize = float(pMax - pMin) / (num_steps - 1)
        return np.arange(pMin, pMax + stepsize, stepsize)

    def polyfit_sim_freq(self, parameter, pMin, pMax, num_steps, parameters_update):
        """Return the relation between a parameter and the spiking frequency.

        Runs simulations for parameter values between pMin and pMax.
        Calculates the spiking frequency f for each simulation.
        Fits a 2nd order polynomial on the resulting dependency of the parameter from f.

        Simulation parameters which are not in args have a predefined value.

        Args:
            parameter (string): parameter which will be sweeped
            pMin: minimum value for parameter
            pMax: maximum value for parameter
            num_steps: number of steps between pMin and pMax
            EL: constant value EL
            Vreset: constant value Vreset
            Vt: constant value Vt

        Returns:
            polynomial coefficients a, b, c for a*x**2 + b*x + c
        """

        freq = []  # frequency array
        values = self.parameter_range(pMin, pMax, num_steps)

        parameters = config.get_parameters(parameter)
        parameters.update(parameters_update)

        simtime = config.get_time(parameter)

        # calculate frequency for each value
        for i in values:
            parameters[parameter] = i
            t, v, spikes = self.run_simulation(simtime, 0, parameters)
            freqs = self.compute_frequencies(spikes)
            freq.append(freqs[0])

        return np.polyfit(freq, values, 2)

    def get_gl_freq(self, gLMin, gLMax, EL, Vreset, Vt):
        """Return the relation between gL and frequency.

        Runs simulations for values of gL between gLMin and gLMax.
        Calculates the spiking frequency f for each simulation.
        Fits a 2nd order polynomial on the resulting gL(f).

        Simulation parameters which are not in args have a predefined value.

        Args:
            gLMin: minimum value for gL
            gLMax: maximum value for gL
            EL: constant parameter EL
            Vreset: constant parameter Vreset
            Vt: constant parameter Vt

        Returns:
            polynomial coefficients a, b, c for a*x**2 + b*x + c
        """

        return self.polyfit_sim_freq("gL", gLMin, gLMax, 4, {"EL": EL, "Vreset": Vreset, "Vt": Vt})

    def get_a_freq(self, paramMin, paramMax, EL, Vreset, Vt, gL):
        """Return the relation between a and frequency"""

        return self.polyfit_sim_freq("a", paramMin, paramMax, 4, {"EL": EL, "Vreset": Vreset, "Vt": Vt, "gL": gL})

    def get_vexp_freq(self, paramMin, paramMax, EL, Vreset, Vt, gL, dT):
        """Return the relation between a and frequency"""

        freq = []  # Frequency array
        values = self.parameter_range(paramMin, paramMax, 20)

        parameters = config.get_parameters("Vexp")
        parameters.update({"EL": EL,
                           "gL": gL,
                           "Vt": Vt,
                           "Vreset": Vreset,
                           "dT": dT})

        simtime = config.get_time("Vexp")

        for i in values:
            parameters["Vexp"] = i
            freq.append(self.get_stat_freq(simtime, 0, parameters))

        # Cut constant part
        thresh = 10e3
        new_freq = []
        new_values = []
        for i in range(1, len(freq)):
            if (math.fabs(freq[i] - freq[i - 1]) > thresh):
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

        return np.polyfit(new_freq, new_values, 2)

    def get_b_freq(self, paramMin, paramMax, EL, Vreset, Vt, gL, a, tauw):
        """Return the relation between b and frequency"""

        return self.polyfit_sim_freq("b", paramMin, paramMax, 4, {"EL": EL, "Vreset": Vreset, "Vt": Vt, "gL": gL, "a": a, "tauw": tauw})

    def get_tauw_isi(self, paramMin, paramMax, EL, Vreset, Vt, gL, gLadapt, stim):
        """Return the relation between tauw and frequency"""

        ISI = []  # Frequency array
        values = self.parameter_range(paramMin, paramMax, 10)

        parameters = config.get_parameters("tauw_isi")
        parameters.update({"EL": EL,
                           "gL": gL,
                           "Vt": Vt,
                           "Vreset": Vreset,
                           "a": gLadapt})

        simtime = config.get_time("tauw_isi")

        for i in values:
            parameters["tw"] = i
            t, v, spikes = self.run_simulation(simtime, stim, parameters)
            freqs = []
            for i in range(1, len(spikes)):
                freqs.append(1. / (spikes[i] - spikes[i - 1]))
            a, b = np.polyfit(range(len(freqs)), freqs, 1)
            ISI.append(b)
            #ISI.append(spikes[1]-spikes[0])

        # for i in range(len(values)):
            # values[i] = float(1/float(values[i]))

        return np.polyfit(ISI, values, 4)

    def get_tauw_int(self, paramMin, paramMax, EL, Vreset, Vt, gL, gLadapt, stim):
        """Return the relation between tauw and ??.

        Runs simulations for values of tauw between paramMin and paramMax.
        Integrates t(V) using the trapezoidal rule for each simulation.
        Fits a 2nd order polynomial on tauw(int(t(V))).

        Returns:
            polynomial coefficients.
        """
        # TODO why integrate t(V)?

        integral = []

        values = self.parameter_range(paramMin, paramMax, 10)

        parameters = config.get_parameters("tauw_int")
        parameters.update({"EL": EL,
                           "gL": gL,
                           "Vt": Vt,
                           "Vreset": Vreset,
                           "a": gLadapt})

        simtime = config.get_time("tauw_int")

        for i in values:
            parameters["tw"] = i
            t, v, spikes = self.run_simulation(simtime, stim, parameters)
            integral.append(np.trapz(t, v))

        return np.polyfit(integral, values, 2)