import pickle
from pycairo.calibration.base import BaseCalibration
from pycairo.helpers.units import Voltage, Current, DAC
import pycairo.logic.spikes
import pyhalbe
from collections import defaultdict
import numpy as np
import os
import time

from parameters import parameters

neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

E_syni_dist = parameters["E_syni_dist"]
E_synx_dist = parameters["E_synx_dist"]

E_l_range = parameters["E_l_range"]
V_t_range = parameters["V_t_range"]
V_reset_range = parameters["V_t_range"]
I_gl_range = parameters["I_gl_range"]

repetitions = parameters["repetitions"]

save_traces = parameters["save_traces"]
save_results = parameters["save_results"]

E_l_description = parameters["E_l_description"]
V_t_description = parameters["V_t_description"]
V_reset_description = parameters["V_reset_description"]
I_gl_description = parameters["I_gl_description"]

folder = parameters["folder"]

class Calibrate_E_l(BaseCalibration):
    """E_l calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_E_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.I_gl: Current(1000),
                neuron_parameter.V_t: Voltage(1000),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_l, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(300)
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

    def process_trace(self, t, v):
        if np.std(v)>50:
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
                neuron_parameter.E_l: Voltage(1200, apply_calibration = True),  # TODO apply calibration?
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

    def process_trace(self, t, v):
        if np.std(v)*1000<5:
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_trace.p"), 'wb'))
            self.logger.WARN("Bad trace with std of only {} mV".format(np.std(v)))
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
                neuron_parameter.E_l: Voltage(800, apply_calibration=True),
                neuron_parameter.I_gl: Current(1000),
                neuron_parameter.V_t: Voltage(600, apply_calibration=True),
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

    def process_trace(self, t, v):
        if np.std(v)*1000<5:
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_trace.p"), 'wb'))
            self.logger.WARN("Bad trace with std of only {} mV".format(np.std(v)))
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
                neuron_parameter.E_l: Voltage(800, apply_calibration = True),
                neuron_parameter.V_t: Voltage(700, apply_calibration = True),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_g_L, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(500, apply_calibration = True)
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
        super(Calibrate_g_L, self).process_calibration_results(neuron_ids, neuron_parameter.I_gl)

    def store_results(self):
        # TODO detect and store broken neurons
        super(Calibrate_g_L, self).store_calibration_results(neuron_parameter.I_gl)


class Test_E_l(BaseCalibration):
    """E_l calibration."""
    def get_parameters(self):
        parameters = super(Test_E_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.I_gl: Current(1000, apply_calibration = True),
                neuron_parameter.V_t: Voltage(1000, apply_calibration = True),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_E_l, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(300, apply_calibration = True)
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

    def process_trace(self, t, v):
        if np.std(v)*1000>50:
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_trace.p"), 'wb'))
            #raise ValueError
        return np.mean(v)*1000 # Get the mean value * 1000 for mV

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass


class Test_V_t(BaseCalibration):
    """V_t calibration."""
    def get_parameters(self):
        parameters = super(Test_V_t, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(1200, apply_calibration = True),  # TODO apply calibration?
                neuron_parameter.I_gl: Current(1000),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_V_t, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(500, apply_calibration = True)
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

    def process_trace(self, t, v):
        if np.std(v)*1000<5:
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_trace.p"), 'wb'))
            #raise ValueError
        return np.max(v)*1000  # Get the mean value * 1000 for mV

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass


class Test_V_reset(BaseCalibration):
    """V_reset calibration."""
    def get_parameters(self):
        parameters = super(Test_V_reset, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(1100, apply_calibration=True),
                neuron_parameter.I_gl: Current(1000),
                neuron_parameter.V_t: Voltage(900, apply_calibration=True),
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

    def process_trace(self, t, v):
        if np.std(v)*1000<5:
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_trace.p"), 'wb'))
            #raise ValueError
        return np.min(v)*1000  # Get the mean value * 1000 for mV

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass

# TODO, playground for digital spike measures atm.
class Test_g_L(BaseCalibration):
    def get_parameters(self):
        parameters = super(Test_g_L, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(800, apply_calibration = True),
                neuron_parameter.V_t: Voltage(700, apply_calibration = True),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_g_L, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][shared_parameter.V_reset] = Voltage(500, apply_calibration = True)
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


