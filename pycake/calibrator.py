"""Manages experiments

"""
import numpy as np
import pylogging
from collections import defaultdict
import pyhalbe
import Coordinate
from scipy.optimize import curve_fit
from pycake.helpers.trafos import HWtoDAC

import time

# shorter names
Enum = Coordinate.Enum
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

# TODO find better name
class BaseCalibrator(object):
    """ Takes experiments and a target parameter and turns this into calibration data via a linear fit.
        Does NOT save any data.

    Args:
        target_parameter: neuron_parameter or shared_parameter
        feature: string containing the feature key (e.g. "mean" for E_l or E_syn, "g_l" for I_gl, ...)
                 this is used to extract the results that were given by the analyzers of an experiment
        experiments: list containing all experiments
    """
    logger = pylogging.get("pycake.calibrator")

    def __init__(self, experiments):
        if not isinstance(experiments, list):
            experiments = [experiments]
        self.experiments = experiments

    def get_key(self):
        """
        """
        return 'mean'

    def check_for_same_config(self, measurements):
        """ Checks if measurements have the same config.
            Only the target parameter is allowed to be different.
        """
        return True # TODO implement this

    def get_step_parameters(self, measurement):
        """
        """
        step_parameters = measurement.get_parameter(
                self.target_parameter, measurement.neurons)
        return step_parameters

    def merge_experiments(self):
        """ Merges all experiments into one result dictionary.
            Returns: 
                dictionary containing merged results
                The structure is as follows:
                {neuron1: [(step1, [result1, result2, ...]), (step2, [result1, result2, ...]), ...],
                {neuron2: [(step2, [result1, result2, ...]), (step2, [result1, result2, ...]), ...],
                 ...}
        """
        merged = defaultdict(list)
        for ex in self.experiments:
            step_id = 0
            for m in ex.measurements:
                step_results = ex.results[step_id]
                neuron_parameters = self.get_step_parameters(m)
                for neuron, step_value in neuron_parameters.iteritems():
                    if step_results[neuron] is not None:
                        merged[neuron].append((step_value, step_results[neuron]))
                step_id += 1

        ordered = {}
        for neuron, result_list in merged.iteritems(): 
            neuron_results = defaultdict(list)
            for step, result in result_list:
                neuron_results[step].append(result[self.get_key()])
            neuron_results_list = [(step, result) for step, result in neuron_results.iteritems()]
            ordered[neuron] = neuron_results_list
        return ordered

    def average_over_experiments(self):
        """ Gives average and (trial-to-trial) standard deviation over all experiments.

            Returns:
                Two dictionaries that are structured as follows:
                {neuron1: [(step1, mean1), (step2, mean2), ...],
                 neuron2: ....}
                and
                {neuron1: [(step1, std1), (step2, std2), ...],
                 neuron2: ....}
        """
        merged = self.merge_experiments()
        mean = {}
        std = {}
        for neuron, all_results in merged.iteritems():
            neuron_mean = []
            neuron_std = []
            for step, results in all_results:
                if len(results) == 0: # If no result at all is found, None should be returned
                    neuron_mean.append((step, None))
                    neuron_std.append((step, None))
                    continue
                mean_result = np.mean(results)
                std_result = np.std(results)
                neuron_mean.append((step, mean_result))
                neuron_std.append((step, std_result))
            mean[neuron] = neuron_mean
            std[neuron] = neuron_std
        return mean, std

    def generate_coeffs(self):
        """ Takes averaged experiments and does the fits
        """
        average, std = self.average_over_experiments()
        coeffs = {}
        for neuron, results in average.iteritems():
            # Need to switch y and x in order to get the right fit
            # (y-axis: configured parameter, x-axis: measurement)
            ys_raw, xs_raw = zip(*results) # 'unzip' results, e.g.: (100,0.1),(200,0.2) -> (100,200), (0.1,0.2)
            xs = self.prepare_x(xs_raw)
            ys = self.prepare_y(ys_raw)
            coeffs[neuron] = self.do_fit(xs, ys)
            if self.is_defect(coeffs[neuron]):
                coeffs[neuron] = None
        return [(self.target_parameter, coeffs)]

    def dac_to_si(self, dac, parameter):
        """ Transforms dac value to mV or nA, depending on input parameter
        """
        if self.target_parameter.name[0] == 'I':
            return dac * 2500/1023.
        else:
            return dac * 1800/1023.

    def prepare_x(self, x):
        """ Prepares x values for fit
            Usually, this is a measured membrane voltage in V
            Per default, this function Translates from V or A to DAC
        """
        xs = [HWtoDAC(val*1000, self.target_parameter) for val in x]
        xs = np.array(xs)
        return xs

    def prepare_y(self, y):
        """ Prepares y values for fit
            Per default, these are the step (DAC) values that were set.
        """
        ys = np.array(y)
        return ys

    def do_fit(self, xs, ys):
        """ Fits a curve to results of one neuron
            Standard behaviour is a linear fit.
        """
        fit_coeffs = np.polyfit(xs, ys, 1)
        return fit_coeffs

    def is_defect(self, coeffs):
        return False 

    def get_neurons(self):
        return self.experiments[0].measurements[0].neurons


class V_reset_Calibrator(BaseCalibrator):
    target_parameter = shared_parameter.V_reset

    def __init__(self, experiment):
        self.experiment = experiment

    def get_key(self):
        return 'baseline'

    def iter_neuron_results(self):
        key = self.get_key()
        for neuron in self.get_neurons():
            baseline, = self.experiment.get_mean_results(
                    neuron, self.target_parameter, (key, ))
            yield neuron, baseline

    def mean_over_blocks(self, neuron_results):
        blocks = defaultdict(list)
        for neuron, value in neuron_results.iteritems():
            blocks[neuron.sharedFGBlock()].append(value)
        return dict((k, np.mean(v, axis=0)) for k, v in blocks.iteritems())

    def get_neurons(self):
        return self.experiment.measurements[0].neurons

    def do_fit(self, raw_x, raw_y):
        print raw_x, raw_y
        xs = self.prepare_x(raw_x)
        ys = self.prepare_y(raw_y)
        return np.polyfit(xs, ys, 1)

    def get_neuron_shifts(self, neuron_results, block_means):
        block_means = self.mean_over_blocks(neuron_results)

        neuron_shifts = {}
        for neuron, value in neuron_results.iteritems():
            diffs = value[:,2] - block_means[neuron.sharedFGBlock()][:,2]
            neuron_shifts[neuron] = [np.mean(diffs)]
        return neuron_shifts

    def generate_coeffs(self):
        """ Takes averaged experiments and does the fits
        """
        neuron_results = dict(x for x in self.iter_neuron_results())
        block_means = self.mean_over_blocks(neuron_results)

        coeffs = {}
        for coord, mean in block_means.iteritems():
            print mean
            coeffs[coord] = self.do_fit(mean[:,2], mean[:,0])

        coeffs.update(self.get_neuron_shifts(neuron_results, block_means))
        return [(self.target_parameter, coeffs)]

class E_synx_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.E_synx
    
    def is_defect(self, coeffs):
        defect = abs(coeffs[0] - 1) > 0.4 # Broken if slope of the fit is too high or too small
        return defect

class E_syni_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.E_syni
    pass

class E_l_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.E_l
    pass

class V_t_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.V_t
    def get_key(self):
        return 'max'

class I_gl_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.I_gl
    def prepare_x(self, x):
        xs = [HWtoDAC(val, self.target_parameter) for val in x]
        xs = np.array(xs)
        return xs

    def get_key(self):
        return 'g_l'

from pycake.calibration.E_l_I_gl_fixed import E_l_I_gl_fixed_Calibrator

class V_syntc_psp_max_BaseCalibrator(BaseCalibrator):

    # to be set in derived class
    target_parameter = None

    def __init__(self, experiments):

        from pycake.experiment import BaseExperiment

        if not isinstance(experiments, list):
            experiments = [experiments]

        self.experiment = BaseExperiment(sum(
            [exp.measurements for exp in experiments], []),
            experiments[0].analyzer, None)
        self.experiment.results = sum(
            [exp.results for exp in experiments], [])
        self.neurons = self.experiment.measurements[0].neurons

    def get_key(self):
        return 'std'

    def fit_neuron(self, neuron):
        data = self.experiment.get_parameters_and_results(neuron,
                                                          [self.target_parameter],
                                                          ["std"])
        index_V_syntc = 0
        index_std = 1

        V_syntc_steps = np.unique(data[:,index_V_syntc])

        V_syntc_psp_max = 0
        max_std = 0

        for step in V_syntc_steps:
            selected = data[data[:,index_V_syntc] == step]
            V_syntc, std = selected[:,(index_V_syntc, index_std)].T

            #V_syntc = V_syntc[0]
            #std = std[0]

            if std > max_std:
                V_syntc_psp_max = V_syntc[0]
                max_std = std[0]

        return V_syntc_psp_max

    def generate_coeffs(self):

        coeffs = {}

        for neuron in self.neurons:
            V_syntc_psp_max = self.fit_neuron(neuron)
            coeffs[neuron] = [V_syntc_psp_max]

        print coeffs

        return [(self.target_parameter, coeffs)]

class V_syntci_psp_max_Calibrator(V_syntc_psp_max_BaseCalibrator):
    target_parameter = neuron_parameter.V_syntci

class V_syntcx_psp_max_Calibrator(V_syntc_psp_max_BaseCalibrator):
    target_parameter = neuron_parameter.V_syntcx
