#/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calibration of synapse parameters."""

import os
import time
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

import pysthal

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

import scipy.stats as stats

class Calibrate_E_synx(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_E_synx, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.I_gl: Current(0),
                neuron_parameter.V_t: Voltage(1700, apply_calibration = True),
                neuron_parameter.I_convi: Current(0),
                neuron_parameter.I_convx: Current(2500),
                neuron_parameter.V_syntcx: Voltage(1800),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_synx, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(200, apply_calibration = True)
        return parameters

    
    def get_steps(self):
        steps = []
        for E_syn_voltage in range(650,1050,50): # 4 steps
            steps.append({neuron_parameter.E_synx: Voltage(E_syn_voltage),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_E_synx, self).init_experiment()
        self.repetitions = 2
        self.save_results = True
        self.save_traces = False
        self.description = "Calibrate_E_synx via synaptic leakage."

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>1000:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Is neuron spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        return np.mean(v) * 1000 
    
    def process_results(self, neuron_ids):
        #pass
        super(Calibrate_E_synx, self).process_calibration_results(neuron_ids, neuron_parameter.E_synx, linear_fit=True)

    def store_results(self):
        #pass
        super(Calibrate_E_synx, self).store_calibration_results(neuron_parameter.E_synx)

class Calibrate_E_syni(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_E_syni, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.I_gl: Current(0),
                neuron_parameter.V_t: Voltage(1700, apply_calibration = True),
                neuron_parameter.I_convi: Current(2500),
                neuron_parameter.I_convx: Current(0),
                neuron_parameter.V_syntci: Voltage(1800),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_syni, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(200, apply_calibration = True)
        return parameters

    
    def get_steps(self):
        steps = []
        for E_syn_voltage in range(350,750,25): # 4 steps
            steps.append({neuron_parameter.E_syni: Voltage(E_syn_voltage),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_E_syni, self).init_experiment()
        self.repetitions = 2
        self.save_results = True
        self.save_traces = False
        self.description = "Calibrate_E_syni via synaptic leakage."

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>1000:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Is neuron spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        return np.mean(v) * 1000 
    
    def process_results(self, neuron_ids):
        super(Calibrate_E_syni, self).process_calibration_results(neuron_ids, neuron_parameter.E_syni, linear_fit=True)

    def store_results(self):
        super(Calibrate_E_syni, self).store_calibration_results(neuron_parameter.E_syni)

