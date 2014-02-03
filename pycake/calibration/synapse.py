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
    target_parameter = neuron_parameter.E_synx

    def init_experiment(self):
        super(Calibrate_E_synx, self).init_experiment()
        self.E_syni_dist = None
        self.E_synx_dist = None
        self.sthal.stimulateNeurons(5.0e6, 4)

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>1000:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Is neuron spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        return np.mean(v) * 1000 
    
    def isbroken(self, coeffs):
        if abs(coeffs[1] - 1) > 0.4:     # Broken if slope of the fit is too high or too small
            return True
        else:
            return False

class Calibrate_E_syni(BaseCalibration):
    target_parameter = neuron_parameter.E_syni

    def init_experiment(self):
        super(Calibrate_E_syni, self).init_experiment()
        self.E_syni_dist = None
        self.E_synx_dist = None

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>1000:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Is neuron spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        return np.mean(v) * 1000 
    
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
    target_parameter = neuron_parameter.E_synx

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>1000:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Is neuron spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        return np.mean(v) * 1000 
    
class Test_E_syni(BaseTest):
    target_parameter = neuron_parameter.E_syni

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>1000:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Is neuron spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        return np.mean(v) * 1000 
