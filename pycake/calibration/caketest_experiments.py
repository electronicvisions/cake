import pickle
from pycake.calibration.base import BaseCalibration
from pycake.helpers.units import Voltage, Current, DAC
from pycake.helpers.trafos import HWtoDAC, HCtoDAC, DACtoHC, DACtoHW
from pycake.helpers.calibtic import create_pycalibtic_polynomial
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
V_reset_range = parameters["V_reset_range"]
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
global_params, global_shared = calibration_parameters["global_params"], calibration_parameters["global_shared"]

test_parameters = parameters["test_params"]
E_l_test_params, E_l_test_shared = test_parameters["E_l_params"], test_parameters["E_l_shared"]
E_syn_test_params, E_syn_test_shared = test_parameters["E_syn_params"], test_parameters["E_syn_shared"]
V_t_test_params, V_t_test_shared = test_parameters["V_t_params"], test_parameters["V_t_shared"]
V_reset_test_params, V_reset_test_shared = test_parameters["V_reset_params"], test_parameters["V_reset_shared"]
g_l_test_params, g_l_test_shared = test_parameters["g_l_params"], test_parameters["g_l_shared"]
global_test_params, global_test_shared = test_parameters["global_params"], test_parameters["global_shared"]

class Calibrate_E_l(pycake.calibration.lif.Calibrate_E_l):
    """E_l calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_E_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(global_params)
            parameters[neuron_id].update(E_l_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_l, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(global_shared)
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
            parameters[neuron_id].update(global_params)
            parameters[neuron_id].update(V_t_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_V_t, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(global_shared)
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
            parameters[neuron_id].update(global_params)
            parameters[neuron_id].update(V_reset_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_V_reset, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(global_shared)
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
            parameters[neuron_id].update(global_params)
            parameters[neuron_id].update(E_syn_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_synx, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(global_shared)
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
            parameters[neuron_id].update(global_params)
            parameters[neuron_id].update(E_syn_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_syni, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(global_shared)
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
            parameters[neuron_id].update(global_params)
            parameters[neuron_id].update(g_l_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_g_L, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(global_shared)
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

class Calibrate_V_reset_shift(pycake.calibration.lif.Calibrate_V_reset):
    """V_reset_shift calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_V_reset_shift, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(global_params)
            parameters[neuron_id].update(V_reset_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_V_reset_shift, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(global_shared)
            parameters[block_id].update(V_reset_shared)
        return parameters

    def get_shared_steps(self):
        steps = []
        for voltage in V_reset_range:
            steps.append({shared_parameter.V_reset: Voltage(voltage, apply_calibration = True)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_V_reset_shift, self).init_experiment()
        localtime = time.localtime()
        self.folder = os.path.join(folder, "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min))
        self.repetitions = repetitions
        self.save_results = save_results
        self.save_traces = save_traces
        self.E_syni_dist = E_syni_dist
        self.E_synx_dist = E_synx_dist
        self.description = "Calibrate_V_reset_shift: {}".format(V_reset_description)

    def process_results(self, neuron_ids):
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
                    results_mean[neuron_id].append(HWtoDAC(np.mean(step_results[neuron_id]), shared_parameter.V_reset))
                    results_std[neuron_id].append(HWtoDAC(np.std(step_results[neuron_id]), shared_parameter.V_reset))
                repetition = 1
                step_results = defaultdict(list)
        
        # Find the shift of each neuron compared to applied V_reset:
        all_steps = self.get_shared_steps()
        for neuron_id in neuron_ids:
            results_mean[neuron_id] = np.array(results_mean[neuron_id])
            results_std[neuron_id] = np.array(results_std[neuron_id])
            steps = [HCtoDAC(step[shared_parameter.V_reset].value, shared_parameter.V_reset) for step in all_steps[neuron_id]]
            # linear fit
            m,b = np.polyfit(steps, results_mean[neuron_id], 1)
            coeffs = [b, m-1]
            # TODO find criteria for broken neurons. Until now, no broken neurons exist
            if self.isbroken(coeffs):
                results_broken.append(neuron_id)
                self.logger.INFO("Neuron {0} marked as broken with coefficients of {1:.2f} and {2:.2f}".format(neuron_id, m, b))
            self.logger.INFO("Neuron {} calibrated successfully with coefficients {}".format(neuron_id, coeffs))
            results_polynomial[neuron_id] = create_pycalibtic_polynomial(coeffs)

        self.results_mean = results_mean
        self.results_std = results_std

        # make final results available
        self.results_polynomial = results_polynomial
        self.results_broken = results_broken

    def store_results(self):
        """This base class function can be used by child classes as store_results."""
        results = self.results_polynomial
        md = cal.MetaData()
        md.setAuthor("pycake")
        md.setComment("calibration")

        logger = self.logger

        collection = self._calib_nc

        nrns = self._red_nrns

        for index in results:
            coord = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(index))
            broken = index in self.results_broken
            if broken:
                if nrns.has(coord):  # not disabled
                    nrns.disable(coord)
                    # TODO reset/delete calibtic function for this neuron
            else:  # store in calibtic
                if not collection.exists(index):
                    logger.INFO("No existing calibration data for neuron {} found, creating default dataset".format(index))
                    calib = cal.NeuronCalibration()
                    collection.insert(index, calib)
                collection.at(index).reset(21, results[index])

        self.logger.INFO("Storing calibration results")
        self.store_calibration(md)



class Test_E_l(pycake.calibration.lif.Calibrate_E_l):
    """E_l calibration."""
    def get_parameters(self):
        parameters = super(Test_E_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(global_test_params)
            parameters[neuron_id].update(E_l_test_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_E_l, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(global_test_shared)
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
            parameters[neuron_id].update(global_test_params)
            parameters[neuron_id].update(V_t_test_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_V_t, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(global_test_shared)
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
            parameters[neuron_id].update(global_test_params)
            parameters[neuron_id].update(V_reset_test_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_V_reset, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(global_test_shared)
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
            parameters[neuron_id].update(global_test_params)
            parameters[neuron_id].update(g_l_test_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_g_L, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(global_test_shared)
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
        parameters = super(Test_E_synx, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(global_test_params)
            parameters[neuron_id].update(E_syn_test_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_E_synx, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(global_test_shared)
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
        parameters = super(Test_E_syni, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update(global_test_params)
            parameters[neuron_id].update(E_syn_test_params)
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_E_syni, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id].update(global_test_shared)
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
