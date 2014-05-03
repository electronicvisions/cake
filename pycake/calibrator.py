"""Manages experiments

"""
import numpy as np
import pylogging
from collections import defaultdict
import pyhalbe
from scipy.optimize import curve_fit

from pycake.helpers.trafos import HWtoDAC

import time

# shorter names
Coordinate = pyhalbe.Coordinate
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
        step_parameters = measurement.get_parameter(self.target_parameter, measurement.neurons)
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
        return coeffs

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

    def get_neurons(self):
        return self.experiments[0].measurements[0].neurons


class V_reset_Calibrator(BaseCalibrator):
    target_parameter = shared_parameter.V_reset

    def get_key(self):
        return 'baseline'

    def get_step_parameters(self, measurement):
        """
        """
        # TODO improve
        neurons = measurement.neurons
        blocks = [Coordinate.FGBlockOnHICANN(Coordinate.Enum(i)) for i in range(4)]
        step_parameters = measurement.get_parameter(self.target_parameter, blocks)
        neuron_parameters = {neuron: step_parameters[neuron.neuronFGBlock()] for neuron in neurons} 
        return neuron_parameters

    def iter_V_reset(self, block):
        neuron_on_quad = Coordinate.NeuronOnQuad(block.x(), block.y())
        for quad in Coordinate.iter_all(Coordinate.QuadOnHICANN):
            yield Coordinate.NeuronOnHICANN(quad, neuron_on_quad)
        return

    def mean_over_blocks(self, averages):
        block_mean = {}

        for block in Coordinate.iter_all(Coordinate.FGBlockOnHICANN):
            neurons_on_block = [n for n in self.iter_V_reset(block)]
            values = [averages[neuron] for neuron in neurons_on_block]
            block_mean[block] = np.mean(values, axis = 0)

        return block_mean 

    def get_neuron_shifts(self, averages):
        neurons = self.get_neurons()
        block_means = self.mean_over_blocks(averages)

        neuron_shifts = {}

        for neuron in neurons:
            block = neuron.sharedFGBlock()
            block_mean = np.array(block_means[block])
            neuron_results = np.array(averages[neuron])
            diffs = neuron_results - block_mean # TODO: check if this is the right way
            mean_diff = np.mean(diffs, axis=0)[1]
            neuron_shifts[neuron] = mean_diff

        return neuron_shifts

    def generate_coeffs(self):
        """ Takes averaged experiments and does the fits
        """
        average, std = self.average_over_experiments()
        block_means = self.mean_over_blocks(average)
        shifts = self.get_neuron_shifts(average)

        coeffs = {}
        for neuron, shift in shifts.iteritems():
            coeffs[neuron] = [shift]
        for block, results in block_means.iteritems():
            # Need to switch y and x in order to get the right fit
            # (y-axis: configured parameter, x-axis: measurement)
            ys_raw, xs_raw = zip(*results) # 'unzip' results, e.g.: (100,0.1),(200,0.2) -> (100,200), (0.1,0.2)
            ys = self.prepare_y(ys_raw)
            xs = self.prepare_x(xs_raw)
            coeffs[block] = self.do_fit(xs, ys)
        return coeffs

class E_synx_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.E_synx
    pass

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
