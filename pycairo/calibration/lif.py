#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calibration of LIF parameters."""

import numpy as np
import pylogging
from collections import defaultdict
import pyhalbe
import pycalibtic
import pyredman as redman
import pycairo.logic.spikes
from pycairo.helpers.calibtic import create_pycalibtic_polynomial
from pycairo.helpers.sthal import StHALContainer, UpdateAnalogOutputConfigurator
from pycairo.helpers.units import Current, Voltage, DAC

# Import everything needed for saving:
import pickle
import time
import os

# shorter names
Coordinate = pyhalbe.Coordinate
Enum = Coordinate.Enum
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter


from pycairo.calibration.base import BaseCalibration

class Calibrate_E_l(BaseCalibration):
    """E_l calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_E_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.I_gl: Current(1000),
                neuron_parameter.V_t: Voltage(1200),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_l, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(500)
        return parameters

    def get_steps(self):
        steps = []
        for voltage in range(400, 700, 50):  # 8 steps
            steps.append({neuron_parameter.E_l: Voltage(voltage),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_E_l, self).init_experiment()
        self.repetitions = 5
        self.save_results = True
        self.save_traces = False
        self.E_syni_dist = -100
        self.E_synx_dist = +100
        self.description = "Calibrate_E_l with Esyn set AFTER calibration."

    def process_trace(self, t, v):
        if np.std(v)*1000>50:
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_trace.p"), 'wb'))
            raise ValueError
        return np.mean(v)*1000 # Get the mean value * 1000 for mV

    def process_results(self, neuron_ids):
        super(Calibrate_E_l, self).process_calibration_results(neuron_ids, neuron_parameter.E_l, linear_fit=True)

    def store_results(self):
        super(Calibrate_E_l, self).store_calibration_results(neuron_parameter.E_l)


class Calibrate_V_t(BaseCalibration):
    """V_t calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_V_t, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(1000, apply_calibration = True),  
                neuron_parameter.I_gl: Current(1000),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_V_t, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(300)
        return parameters

    def get_steps(self):
        steps = []
        for voltage in range(500, 900, 50): # 8 steps
            steps.append({neuron_parameter.V_t: Voltage(voltage)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_V_t, self).init_experiment()
        self.repetitions = 5
        self.save_results = True
        self.save_traces = False
        self.E_syni_dist = -100
        self.E_synx_dist = +100
        self.description = "Calibrate_V_t with 1000 I_pl. Calibrated E_l."

    def process_trace(self, t, v):
        return np.max(v)*1000  # Get the mean value * 1000 for mV

    def process_results(self, neuron_ids):
        super(Calibrate_V_t, self).process_calibration_results(neuron_ids, neuron_parameter.V_t)

    def store_results(self):
        super(Calibrate_V_t, self).store_calibration_results(neuron_parameter.V_t)


class Calibrate_V_reset(BaseCalibration):
    """V_reset calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_V_reset, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(1100, apply_calibration=True),
                neuron_parameter.I_gl: Current(1000),
                neuron_parameter.V_t: Voltage(900, apply_calibration=True),
                neuron_parameter.I_convi: DAC(1023),
                neuron_parameter.I_convx: DAC(1023),
            })
        # TODO apply V_t calibration?
        return parameters

    def get_shared_steps(self):
        steps = []
        for voltage in range(300, 500, 25): # 8 steps
            steps.append({shared_parameter.V_reset: Voltage(voltage)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_V_reset, self).init_experiment()
        self.repetitions = 5
        self.save_results = True
        self.save_traces = False
        self.E_syni_dist = -100
        self.E_synx_dist = +100
        self.description = "Calibrate_V_reset, I_pl HIGH. E_l and V_t calibrated."

    def process_trace(self, t, v):
        return np.min(v)*1000  # Get the mean value * 1000 for mV

    def process_results(self, neuron_ids):
        super(Calibrate_V_reset, self).process_calibration_results(neuron_ids, shared_parameter.V_reset)

    def store_results(self):
        # TODO detect and store broken neurons
        super(Calibrate_V_reset, self).store_calibration_results(shared_parameter.V_reset)


# TODO, playground for digital spike measures atm.
class Calibrate_g_L(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_g_L, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(1100, apply_calibration = True),
                neuron_parameter.V_t: Voltage(1000, apply_calibration = True),
                neuron_parameter.I_convx: DAC(1023),
                neuron_parameter.I_convi: DAC(1023),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_g_L, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(800, apply_calibration = True)
        return parameters

    def get_steps(self):
        steps = []
        #for current in range(700, 1050, 50):
        for current in range(700, 1000, 100):
            steps.append({neuron_parameter.I_gl: Current(current)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_g_L, self).init_experiment()
        self.description = "Capacitance g_L experiment." # Change this for all child classes
        self.repetitions = 1
        self.E_syni_dist = -100
        self.E_synx_dist = +100
        self.save_results = True
        self.save_traces = False

    def process_trace(self, t, v):
        spk = pycairo.logic.spikes.detect_spikes(t,v)
        f = pycairo.logic.spikes.spikes_to_freqency(spk)
        E_l = 1100.
        C = 2.16456E-12
        V_t = max(v)*1000.
        V_r = min(v)*1000.
        g_l = f * C * np.log((V_r-E_l)/(V_t-E_l)) * 1E9
        return g_l

    def process_calibration_results(self, neuron_ids, parameter, linear_fit=False):
        """This base class function can be used by child classes as process_results."""
        # containers for final results
        results_mean = defaultdict(list)
        results_std = defaultdict(list)
        results_polynomial = {}
        results_broken = []  # will contain neuron_ids which are broken

        # self.all_results contains all measurements, need to untangle
        # repetitions and steps
        # This is done for shared and neuron parameters
        repetition = 1
        step_results = defaultdict(list)
        for result in self.all_results:
            # still in the same step, collect repetitions for averaging
            for neuron_id in neuron_ids:
                step_results[neuron_id].append(result[neuron_id])
            repetition += 1
            if repetition > self.repetitions:
                # step is done; average, store and reset
                for neuron_id in neuron_ids:
                    results_mean[neuron_id].append(np.mean(step_results[neuron_id]))
                    results_std[neuron_id].append(np.std(step_results[neuron_id]))
                repetition = 1
                step_results = defaultdict(list)


        all_steps = self.get_steps()
        for neuron_id in neuron_ids:
            results_mean[neuron_id] = np.array(results_mean[neuron_id])
            results_std[neuron_id] = np.array(results_std[neuron_id])
            steps = [step[parameter].value for step in all_steps[neuron_id]]
            if linear_fit:
                # linear fit
                m, b = np.linalg.lstsq(zip(results_mean[neuron_id], [1]*len(results_mean[neuron_id])), steps)[0]
                coeffs = [b, m]
                if parameter is neuron_parameter.E_l:
                    if not (m > 1.0 and m < 1.5 and b < 0 and b > -500):
                        # this neuron is broken
                        results_broken.append(neuron_id)
            else:
                # fit polynomial to results
                weight = 1./(results_std[neuron_id] + 1e-8)  # add a tiny value because std may be zero
                # note that np.polynomial.polynomial.polyfit coefficients have
                # reverse order compared to np.polyfit
                # to reverse coefficients: rcoeffs = coeffs[::-1]
                coeffs = np.polynomial.polynomial.polyfit(results_mean[neuron_id], steps, 2, w=weight)
                # TODO check coefficients for broken behavior
            results_polynomial[neuron_id] = create_pycalibtic_polynomial(coeffs)

        self.results_mean = results_mean
        self.results_std = results_std

        # make final results available
        self.results_polynomial = results_polynomial
        self.results_broken = results_broken

    def process_results(self, neuron_ids):
        self.process_calibration_results(neuron_ids, neuron_parameter.I_gl)

    def store_results(self):
        # TODO detect and store broken neurons
        super(Calibrate_g_L, self).store_calibration_results(neuron_parameter.I_gl)


# TODO
class Calibrate_tau_ref(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_tau_ref, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(800),
                neuron_parameter.V_t: Voltage(700),
                shared_parameter.V_reset: Voltage(500),
                neuron_parameter.I_gl: Current(1000),
                neuron_parameter.V_syntcx: Voltage(10)
            })
        return parameters

    def measure(self, neuron_ids):
        import pycairo.simulator
        params = self.get_parameters()
        results = {}
        base_freq = pycairo.simulator.Simulator().compute_freq(30e-6, 0, params)

        for neuron_id in neuron_ids:
            self.sthal.switch_analog_output(neuron_id)
            self.sthal.adc.record()
            ts = np.array(self.sthal.adc.getTimestamps())
            v = np.array(self.sthal.adc.trace())

            calc_freq = lambda x: (1.0 / x - 1.0 / base_freq) * 1e6  # millions?

# TODO
#            m = sorted_array_mean[sorted_array_mean != 0.0]
#            e = sorted_array_err[sorted_array_err != 0.0]
#            return calc_freq(m), calc_freq(e)

            results[neuron_id] = calc_freq(v)
            del ts

        self.all_results.append(results)

# TODO
class Calibrate_tau_synx(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_tau_synx, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(600),
                neuron_parameter.E_synx: Voltage(1300),
                neuron_parameter.E_syni: Voltage(200),
                neuron_parameter.V_t: Voltage(1700),
                shared_parameter.V_reset: Voltage(500),
                neuron_parameter.I_gl: Current(1000),
            })
        return parameters

    def measure(self, neuron_ids):
        pass  # TODO

# TODO
class Calibrate_tau_syni(BaseCalibration):
    def measure(self, neuron_ids):
        pass  # TODO

class Calibrate_g_L_stepcurrent(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_g_L_stepcurrent, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                pyhalbe.HICANN.neuron_parameter.E_l: Voltage(900, apply_calibration = True),
                pyhalbe.HICANN.neuron_parameter.V_t: Voltage(1300, apply_calibration = True),
                pyhalbe.HICANN.neuron_parameter.I_convx: DAC(1023),
                pyhalbe.HICANN.neuron_parameter.I_convi: DAC(1023),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_g_L_stepcurrent, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][pyhalbe.HICANN.shared_parameter.V_reset] = Voltage(700, apply_calibration = True)
        return parameters

    def get_steps(self):
        steps = []
        #for current in range(700, 1050, 50):
        for current in range(700, 1000, 100):
            steps.append({pyhalbe.HICANN.neuron_parameter.I_gl: Current(current)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_g_L_stepcurrent, self).init_experiment()
        self.description = "Capacitance g_L experiment. Membrane is fitted after step current" # Change this for all child classes
        self.repetitions = 1
        self.E_syni_dist = -100
        self.E_synx_dist = +100
        self.save_results = True
        self.save_traces = False

    def prepare_measurement(self, step_parameters, step_id, rep_id):
        """Prepare measurement.
        Perform reset, write general hardware settings.
        """
        neuron_parameters = step_parameters[0]
        shared_parameters = step_parameters[1]

        fgc = pyhalbe.HICANN.FGControl()
        # Set neuron parameters for each neuron
        for neuron_id in neuron_parameters:
            coord = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(neuron_id))
            for parameter in neuron_parameters[neuron_id]:
                value = neuron_parameters[neuron_id][parameter]
                fgc.setNeuron(coord, parameter, value)
            fgstim = pyhalbe.HICANN.FGStimulus(10,10,False)
            self.sthal.hicann.setCurrentStimulus(coord, fgstim)

        # Set block parameters for each block
        for block_id in shared_parameters:
            coord = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(block_id))
            for parameter in shared_parameters[block_id]:
                value = shared_parameters[block_id][parameter]
                fgc.setShared(coord, parameter, value)

        self.sthal.hicann.floating_gates = fgc

        if self.bigcap is True:
            # TODO add bigcap functionality
            pass

        if self.save_results:
            if not os.path.isdir(os.path.join(self.folder,"floating_gates/")):
                os.mkdir(os.path.join(self.folder,"floating_gates/"))
            pickle.dump(fgc, open(os.path.join("floating_gates/","step{}rep{}.p".format(step_id,rep_id)), 'wb'))

        self.sthal.write_config()

    def process_trace(self, t, v):
        spk = pycairo.logic.spikes.detect_spikes(t,v)
        f = pycairo.logic.spikes.spikes_to_freqency(spk)
        E_l = 1100.
        C = 2.16456E-12
        V_t = max(v)*1000.
        V_r = min(v)*1000.
        g_l = f * C * np.log((V_r-E_l)/(V_t-E_l)) * 1E9
        return g_l

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass


