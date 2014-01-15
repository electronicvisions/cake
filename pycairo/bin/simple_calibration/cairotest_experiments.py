import pickle
from pycairo.calibration.base import BaseCalibration
from pycairo.helpers.units import Voltage, Current, DAC
import pycairo.logic.spikes
import pyhalbe
from collections import defaultdict
import numpy as np
import os
import time
import pycairo.calibration.lif
import pycairo.calibration.synapse

from parameters import parameters

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


E_l_params = {"V_t":        1200,
              "V_reset":    300,
              "I_gl":       1000,
             }

V_t_params = {"E_l":        1000,
              "V_reset":    400,
              "I_gl":       1000,
             }

V_reset_params = {"E_l":    1100,
                  "V_t":    900,
                  "I_gl":   1000,
                 }

g_l_params = {"E_l":        700,
              "V_t":        600,
              "V_reset":    450,
             }

 

class Calibrate_E_l(pycairo.calibration.lif.Calibrate_E_l):
    """E_l calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_E_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.I_gl: Current(E_l_params["I_gl"]),
                neuron_parameter.V_t: Voltage(E_l_params["V_t"]),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_l, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(E_l_params["V_reset"])
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


class Calibrate_V_t(pycairo.calibration.lif.Calibrate_V_t):
    """V_t calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_V_t, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(V_t_params["E_l"], apply_calibration = True),  # TODO apply calibration?
                neuron_parameter.I_gl: Current(V_t_params["I_gl"]),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_V_t, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(V_t_params["V_reset"])
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


class Calibrate_V_reset(pycairo.calibration.lif.Calibrate_V_reset):
    """V_reset calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_V_reset, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(V_reset_params["E_l"], apply_calibration=True),
                neuron_parameter.I_gl: Current(V_reset_params["I_gl"]),
                neuron_parameter.V_t: Voltage(V_reset_params["V_t"], apply_calibration=True),
            })
        # TODO apply V_t calibration?
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

class Calibrate_E_synx(pycairo.calibration.synapse.Calibrate_E_synx):
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

class Calibrate_E_syni(pycairo.calibration.synapse.Calibrate_E_syni):
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
class Calibrate_g_L(pycairo.calibration.lif.Calibrate_g_L):
    def get_parameters(self):
        parameters = super(Calibrate_g_L, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(g_l_params["E_l"], apply_calibration = True),
                neuron_parameter.V_t: Voltage(g_l_params["V_t"], apply_calibration = True),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_g_L, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(g_l_params["V_reset"], apply_calibration = True)
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


class Test_E_l(pycairo.calibration.lif.Calibrate_E_l):
    """E_l calibration."""
    def get_parameters(self):
        parameters = super(Test_E_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.I_gl: Current(E_l_params["I_gl"], apply_calibration = True),
                neuron_parameter.V_t: Voltage(E_l_params["V_t"], apply_calibration = True),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_E_l, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(E_l_params["V_reset"], apply_calibration = True)
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


class Test_V_t(pycairo.calibration.lif.Calibrate_V_t):
    """V_t calibration."""
    def get_parameters(self):
        parameters = super(Test_V_t, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(V_t_params["E_l"], apply_calibration = True),  # TODO apply calibration?
                neuron_parameter.I_gl: Current(V_t_params["I_gl"]),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_V_t, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(V_t_params["V_reset"], apply_calibration = True)
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


class Test_V_reset(pycairo.calibration.lif.Calibrate_V_reset):
    """V_reset calibration."""
    def get_parameters(self):
        parameters = super(Test_V_reset, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(V_reset_params["E_l"], apply_calibration=True),
                neuron_parameter.I_gl: Current(V_reset_params["I_gl"]),
                neuron_parameter.V_t: Voltage(V_reset_params["V_t"], apply_calibration=True),
            })
        # TODO apply V_t calibration?
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
class Test_g_L(pycairo.calibration.lif.Calibrate_g_L):
    def get_parameters(self):
        parameters = super(Test_g_L, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(g_l_params["E_l"], apply_calibration = True),
                neuron_parameter.V_t: Voltage(g_l_params["V_t"], apply_calibration = True),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_g_L, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(g_l_params["V_reset"], apply_calibration = True)
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

class Test_E_synx(pycairo.calibration.synapse.Calibrate_E_synx):
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

class Test_E_syni(pycairo.calibration.synapse.Calibrate_E_syni):
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
