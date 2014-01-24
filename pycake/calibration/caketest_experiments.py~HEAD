import pickle
from pycake.calibration.base import BaseCalibration
from pycake.helpers.units import Voltage, Current, DAC
import pycake.logic.spikes
import pyhalbe
from collections import defaultdict
import numpy as np
import os
import time
import pycake.calibration.lif
import pycake.calibration.synapse

from parameters import parameters

import pycalibtic as cal

neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

E_syni_dist = parameters["E_syni_dist"]
E_synx_dist = parameters["E_synx_dist"]

E_l_range = parameters["E_l_range"]
V_t_range = parameters["V_t_range"]
V_reset_range = parameters["V_t_range"]
I_gl_range = parameters["I_gl_range"]
E_synx_range = parameters["E_synx_range"]
E_syni_range = parameters["E_syni_range"]

repetitions = parameters["repetitions"]

save_traces = parameters["save_traces"]
save_results = parameters["save_results"]

E_l_description = parameters["E_l_description"]
V_t_description = parameters["V_t_description"]
V_reset_description = parameters["V_reset_description"]
I_gl_description = parameters["I_gl_description"]
E_synx_description = parameters["E_synx_description"]
E_syni_description = parameters["E_syni_description"]

folder = parameters["folder"]

calibration_parameters = parameters["calibration_params"]
E_l_params, E_l_shared = calibration_parameters["E_l_params"], calibration_parameters["E_l_shared"]
E_syn_params, E_syn_shared = calibration_parameters["E_syn_params"], calibration_parameters["E_syn_shared"]
V_t_params, V_t_shared = calibration_parameters["V_t_params"], calibration_parameters["V_t_shared"]
V_reset_params, V_reset_shared = calibration_parameters["V_reset_params"], calibration_parameters["V_reset_shared"]
g_l_params, g_l_shared = calibration_parameters["g_l_params"], calibration_parameters["g_l_shared"]

test_parameters = parameters["test_params"]
E_l_test_params, E_l_test_shared = test_parameters["E_l_params"], test_parameters["E_l_shared"]
E_syn_test_params, E_syn_test_shared = test_parameters["E_syn_params"], test_parameters["E_syn_shared"]
V_t_test_params, V_t_test_shared = test_parameters["V_t_params"], test_parameters["V_t_shared"]
V_reset_test_params, V_reset_test_shared = test_parameters["V_reset_params"], test_parameters["V_reset_shared"]
g_l_test_params, g_l_test_shared = test_parameters["g_l_params"], test_parameters["g_l_shared"]

class Calibrate_E_l(pycake.calibration.lif.Calibrate_E_l):
    """E_l calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_E_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(E_l_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_l, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(E_l_shared)
        return parameters

    def get_steps(self):
        steps = []
        for voltage in E_l_range:
            steps.append({neuron_parameter.E_l: Voltage(voltage),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_E_l, self).init_experiment()
        localtime = time.localtime()
        self.folder = os.path.join(folder, "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min))
        self.repetitions = repetitions
        self.save_results = save_results
        self.save_traces = save_traces
        self.E_syni_dist = E_syni_dist
        self.E_synx_dist = E_synx_dist 
        self.description = "Calibrate E_l: {}".format(E_l_description)


class Calibrate_V_t(pycake.calibration.lif.Calibrate_V_t):
    """V_t calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_V_t, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(V_t_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_V_t, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(V_t_shared)
        return parameters

    def get_steps(self):
        steps = []
        for voltage in V_t_range:
            steps.append({neuron_parameter.V_t: Voltage(voltage)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_V_t, self).init_experiment()
        localtime = time.localtime()
        self.folder = os.path.join(folder, "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min))
        self.repetitions = repetitions
        self.save_results = save_results
        self.save_traces = save_traces 
        self.E_syni_dist = E_syni_dist 
        self.E_synx_dist = E_synx_dist
        self.description = "Calibrate_V_t: ".format(V_t_description)


class Calibrate_V_reset(pycake.calibration.lif.Calibrate_V_reset):
    """V_reset calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_V_reset, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(V_reset_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_V_reset, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(V_reset_shared)
        return parameters

    def get_shared_steps(self):
        steps = []
        for voltage in V_reset_range:
            steps.append({shared_parameter.V_reset: Voltage(voltage)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_V_reset, self).init_experiment()
        localtime = time.localtime()
        self.folder = os.path.join(folder, "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min))
        self.repetitions = repetitions
        self.save_results = save_results
        self.save_traces = save_traces
        self.E_syni_dist = E_syni_dist
        self.E_synx_dist = E_synx_dist
        self.description = "Calibrate_V_reset: {}".format(V_reset_description)

class Calibrate_E_synx(pycake.calibration.synapse.Calibrate_E_synx):
    def get_parameters(self):
        parameters = super(Calibrate_E_synx, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(E_syn_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_synx, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(E_syn_shared)
        return parameters

    def get_steps(self):
        steps = []
        for E_syn_voltage in E_synx_range: 
            steps.append({neuron_parameter.E_synx: Voltage(E_syn_voltage),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_E_synx, self).init_experiment()
        localtime = time.localtime()
        self.folder = os.path.join(folder, "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min))
        self.repetitions = repetitions
        self.save_results = save_results
        self.save_traces = save_traces
        self.description = "Calibrate_E_synx: {}".format(E_synx_description)

class Calibrate_E_syni(pycake.calibration.synapse.Calibrate_E_syni):
    def get_parameters(self):
        parameters = super(Calibrate_E_syni, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(E_syn_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_syni, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(E_syn_shared)
        return parameters

    def get_steps(self):
        steps = []
        for E_syn_voltage in E_syni_range:
            steps.append({neuron_parameter.E_syni: Voltage(E_syn_voltage),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_E_syni, self).init_experiment()
        localtime = time.localtime()
        self.folder = os.path.join(folder, "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min))
        self.repetitions = repetitions
        self.save_results = save_results
        self.save_traces = save_traces
        self.description = "Calibrate_E_syni: {}".format(E_syni_description)
    

# TODO, playground for digital spike measures atm.
class Calibrate_g_L(pycake.calibration.lif.Calibrate_g_L):
    def get_parameters(self):
        parameters = super(Calibrate_g_L, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(g_l_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_g_L, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(g_l_params)
        return parameters

    def get_steps(self):
        steps = []
        for current in I_gl_range:
            steps.append({neuron_parameter.I_gl: Current(current)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_g_L, self).init_experiment()
        localtime = time.localtime()
        self.folder = os.path.join(folder, "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min))
        self.description = "Calibrate_g_L: {}".format(I_gl_description)
        self.repetitions = repetitions
        self.E_syni_dist = E_syni_dist
        self.E_synx_dist = E_synx_dist
        self.save_results = save_results
        self.save_traces = save_traces

    def process_results(self, neuron_ids):
        super(Calibrate_g_L, self).process_calibration_results(neuron_ids, neuron_parameter.I_gl)

    def store_results(self):
        # TODO detect and store broken neurons
        super(Calibrate_g_L, self).store_calibration_results(neuron_parameter.I_gl)


class Test_E_l(pycake.calibration.lif.Calibrate_E_l):
    """E_l calibration."""
    def get_parameters(self):
        parameters = super(Test_E_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(E_l_test_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_E_l, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(E_l_test_shared)
        return parameters

    def get_steps(self):
        steps = []
        for voltage in E_l_range:
            steps.append({neuron_parameter.E_l: Voltage(voltage, apply_calibration = True),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Test_E_l, self).init_experiment()
        localtime = time.localtime()
        self.folder = os.path.join(folder, "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min))
        self.repetitions = repetitions
        self.save_results = save_results
        self.save_traces = save_traces
        self.E_syni_dist = E_syni_dist
        self.E_synx_dist = E_synx_dist
        self.description = "Test E_l: {}".format(E_l_description)

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass


class Test_V_t(pycake.calibration.lif.Calibrate_V_t):
    """V_t calibration."""
    def get_parameters(self):
        parameters = super(Test_V_t, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(V_t_test_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_V_t, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(V_t_test_shared)
        return parameters

    def get_steps(self):
        steps = []
        for voltage in V_t_range: 
            steps.append({neuron_parameter.V_t: Voltage(voltage, apply_calibration = True)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Test_V_t, self).init_experiment()
        localtime = time.localtime()
        self.folder = os.path.join(folder, "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min))
        self.repetitions = repetitions
        self.save_results = save_results
        self.save_traces = save_traces 
        self.E_syni_dist = E_syni_dist
        self.E_synx_dist = E_synx_dist
        self.description = "Test_V_t: {}".format(V_t_description)

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass


class Test_V_reset(pycake.calibration.lif.Calibrate_V_reset):
    """V_reset calibration."""
    def get_parameters(self):
        parameters = super(Test_V_reset, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(V_reset_test_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_V_reset, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(V_reset_test_shared)
        return parameters

    def get_shared_steps(self):
        steps = []
        for voltage in V_reset_range:
            steps.append({shared_parameter.V_reset: Voltage(voltage, apply_calibration = True)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Test_V_reset, self).init_experiment()
        localtime = time.localtime()
        self.folder = os.path.join(folder, "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min))
        self.repetitions = repetitions
        self.save_results = save_results
        self.save_traces = save_traces
        self.E_syni_dist = E_syni_dist
        self.E_synx_dist = E_synx_dist
        self.description = "Test_V_reset: {}".format(V_reset_description)

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass

# TODO, playground for digital spike measures atm.
class Test_g_L(pycake.calibration.lif.Calibrate_g_L):
    def get_parameters(self):
        parameters = super(Test_g_L, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(g_l_test_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_g_L, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(g_l_test_shared)
        return parameters

    def get_steps(self):
        steps = []
        for current in I_gl_range:
            steps.append({neuron_parameter.I_gl: Current(current, apply_calibration = True)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Test_g_L, self).init_experiment()
        localtime = time.localtime()
        self.folder = os.path.join(folder, "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min))
        self.description = "Test_g_L: {}".format(I_gl_description)
        self.repetitions = repetitions
        self.E_syni_dist = E_syni_dist
        self.E_synx_dist = E_synx_dist
        self.save_results = save_results
        self.save_traces = save_traces

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass

class Test_E_synx(pycake.calibration.synapse.Calibrate_E_synx):
    def get_parameters(self):
        parameters = super(Calibrate_E_synx, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(E_syn_test_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_synx, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(E_syn_test_shared)
        return parameters

    def get_steps(self):
        steps = []
        for E_syn_voltage in E_synx_range: # 4 steps
            steps.append({neuron_parameter.E_synx: Voltage(E_syn_voltage, apply_calibration = True),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Test_E_synx, self).init_experiment()
        localtime = time.localtime()
        self.folder = os.path.join(folder, "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min))
        self.repetitions = repetitions
        self.save_results = save_results
        self.save_traces = save_traces
        self.description = "Test_E_synx: {}".format(E_syni_description)

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass

class Test_E_syni(pycake.calibration.synapse.Calibrate_E_syni):
    def get_parameters(self):
        parameters = super(Calibrate_E_syni, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(E_syn_test_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_syni, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(E_syn_test_shared)
        return parameters

    def get_steps(self):
        steps = []
        for E_syn_voltage in E_syni_range: # 4 steps
            steps.append({neuron_parameter.E_syni: Voltage(E_syn_voltage, apply_calibration = True),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Test_E_syni, self).init_experiment()
        localtime = time.localtime()
        self.folder = os.path.join(folder, "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min))
        self.repetitions = repetitions
        self.save_results = save_results
        self.save_traces = save_traces
        self.description = "Test_E_syni: {}".format(E_syni_description)
 
    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass
