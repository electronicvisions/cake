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
import pycake.logic.spikes
import scipy.stats as stats
from pycake.helpers.calibtic import create_pycalibtic_polynomial
from pycake.helpers.sthal import StHALContainer, UpdateAnalogOutputConfigurator
from pycake.helpers.units import Current, Voltage, DAC
from pycake.calibration.base import BaseCalibration, BaseTest

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

class Calibrate_E_synx(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_E_synx, self).get_parameters()
        for neuron_id in self.get_neurons():
            for param, value in self.E_synx_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    parameters[neuron_id][param] = value
                elif isinstance(param, shared_parameter):
                    pass
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_synx, self).get_shared_parameters()
        for block_id in range(4):
            for param, value in self.E_synx_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    pass
                elif isinstance(param, shared_parameter):
                    parameters[block_id][param] = value
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    
    def get_steps(self):
        steps = []
        for E_syn_voltage in self.experiment_parameters["E_synx_range"]: # 4 steps
            steps.append({neuron_parameter.E_synx: Voltage(E_syn_voltage),
                })
        #return {neuron_id: steps for neuron_id in self.get_neurons()}
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_E_synx, self).init_experiment()
        self.description = self.experiment_parameters["E_synx_description"]
        self.E_synx_parameters = self.experiment_parameters['E_synx_parameters']
        self.E_syni_dist = None
        self.E_synx_dist = None

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>1000:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Is neuron spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        return np.mean(v) * 1000 
    
    def process_results(self, neuron_ids):
        super(Calibrate_E_synx, self).process_calibration_results(neuron_ids, neuron_parameter.E_synx, linear_fit=True)

    def store_results(self):
        super(Calibrate_E_synx, self).store_calibration_results(neuron_parameter.E_synx)

    def isbroken(self, coeffs):
        if abs(coeffs[1] - 1) > 0.4:     # Broken if slope of the fit is too high or too small
            return True
        else:
            return False

class Calibrate_E_syni(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_E_syni, self).get_parameters()
        for neuron_id in self.get_neurons():
            for param, value in self.E_syni_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    parameters[neuron_id][param] = value
                elif isinstance(param, shared_parameter):
                    pass
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_syni, self).get_shared_parameters()
        for block_id in range(4):
            for param, value in self.E_syni_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    pass
                elif isinstance(param, shared_parameter):
                    parameters[block_id][param] = value
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_steps(self):
        steps = []
        for E_syn_voltage in self.experiment_parameters["E_syni_range"]: # 4 steps
            steps.append({neuron_parameter.E_syni: Voltage(E_syn_voltage),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_E_syni, self).init_experiment()
        self.description = self.experiment_parameters['E_syni_description']
        self.E_syni_parameters = self.experiment_parameters['E_syni_parameters']
        self.E_syni_dist = None
        self.E_synx_dist = None

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

    def isbroken(self, coeffs):
        if abs(coeffs[1] - 1) > 0.4:     # Broken if slope of the fit is too high or too small
            return True
        else:
            return False


# TODO
class Calibrate_tau_synx(BaseCalibration):
    pass

class Calibrate_tau_syni(BaseCalibration):
    pass





class Test_E_synx(BaseTest):
    def get_parameters(self):
        parameters = super(Test_E_synx, self).get_parameters()
        for neuron_id in self.get_neurons():
            for param, value in self.E_synx_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    parameters[neuron_id][param] = value
                    parameters[neuron_id][param].apply_calibration = True
                elif isinstance(param, shared_parameter):
                    pass
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_E_synx, self).get_shared_parameters()
        for block_id in range(4):
            for param, value in self.E_synx_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    pass
                elif isinstance(param, shared_parameter):
                    parameters[block_id][param] = value
                    parameters[block_id][param].apply_calibration = True
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    
    def get_steps(self):
        steps = []
        for E_syn_voltage in self.experiment_parameters["E_synx_range"]: 
            steps.append({neuron_parameter.E_synx: Voltage(E_syn_voltage, apply_calibration = True),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Test_E_synx, self).init_experiment()
        self.description = "TEST OF " + self.experiment_parameters["E_synx_description"]
        self.E_synx_parameters = self.experiment_parameters['E_synx_parameters']
        self.E_syni_dist = None
        self.E_synx_dist = None

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>1000:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Is neuron spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        return np.mean(v) * 1000 
    
class Test_E_syni(BaseTest):
    def get_parameters(self):
        parameters = super(Test_E_syni, self).get_parameters()
        for neuron_id in self.get_neurons():
            for param, value in self.E_syni_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    parameters[neuron_id][param] = value
                    parameters[neuron_id][param].apply_calibration = True
                elif isinstance(param, shared_parameter):
                    pass
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_E_syni, self).get_shared_parameters()
        for block_id in range(4):
            for param, value in self.E_syni_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    pass
                elif isinstance(param, shared_parameter):
                    parameters[block_id][param] = value
                    parameters[block_id][param].apply_calibration = True
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_steps(self):
        steps = []
        for E_syn_voltage in self.experiment_parameters["E_syni_range"]: # 4 steps
            steps.append({neuron_parameter.E_syni: Voltage(E_syn_voltage, apply_calibration = True),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Test_E_syni, self).init_experiment()
        self.description = "TEST OF " + self.experiment_parameters['E_syni_description']
        self.E_syni_parameters = self.experiment_parameters['E_syni_parameters']
        self.E_syni_dist = None
        self.E_synx_dist = None

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>1000:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Is neuron spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        return np.mean(v) * 1000 


