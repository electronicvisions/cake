import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pycake.calibrationrunner
import Coordinate as C
from bz2 import BZ2File

class Reader(object):
    def __init__(self, runner):
        if isinstance(runner, pycake.calibrationrunner.CalibrationRunner):
            self.runner = runner
        elif os.path.isfile(runner):
            f_open = open
            if runner.endswith('.bz2'):
                f_open = BZ2File
            with f_open(runner, 'rb') as infile:
                self.runner = cPickle.load(infile)
        else:
            print "Not a valid file or runner"

    def get_neurons(self):
        return self.runner.config.get_neurons()


    def get_result(self, parameter, neuron, key, repetition = 0):
        """ Get measurement results for one neuron.

            Args:
                parameter: which parameter?
                neurons: neuron coordinate or id
                key: which key? (e.g. 'mean', 'std', 'baseline', ...)
                repetition: single repetition or mean over all? (mean not yet implemented)

            Returns:
                list of all results
        """
        ex = self.runner.experiments[parameter][repetition]

        if isinstance(neuron, int):
            neuron = C.NeuronOnHICANN(C.Enum(neuron))
        return [r[neuron][key] for r in ex.results]

    def get_results(self, parameter, neurons, key, repetition = 0):
        """ Get measurement results for one neuron.

            Args:
                parameter: which parameter?
                neurons: list of neuron coordinates or ids
                key: which key? (e.g. 'mean', 'std', 'baseline', ...)
                repetition: single repetition or mean over all? (mean not yet implemented)

            Returns:
                {neuron: [results]} if neurons
        """
        ex = self.runner.experiments[parameter][repetition]

        results = {}
        for neuron in neurons:
            if isinstance(neuron, int):
                neuron = C.NeuronOnHICANN(C.Enum(neuron))
            results[neuron] = [r[neuron][key] for r in ex.results]
        return results

    def plot_hist(self, parameter, key, step, repetition=0, **kwargs):
        results = self.get_results(parameter, self.get_neurons(), key, repetition)
        results_list = np.array(results.values())[:,step]
        return plt.hist(results_list, **kwargs)



