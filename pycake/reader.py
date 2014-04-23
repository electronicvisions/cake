import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pycake.calibrationrunner
import Coordinate as C

class Reader(object):
    def __init__(self, runner):
        if isinstance(runner, pycake.calibrationrunner.CalibrationRunner):
            self.runner = runner
        elif os.path.isfile(runner):
            self.runner = pickle.load(open(runner))
        else:
            print "Not a valid file or runner"

    def get_neurons(self):
        return self.runner.config.get_neurons()


    def get_results(self, parameter, neurons, key, repetition = 0):
        """ Get measurement results for one neuron.

            Args:
                parameter: which parameter?
                neurons: neuron coordinate (list) or ids
                key: which key? (e.g. 'mean', 'std', 'baseline', ...)
                repetition: single repetition or mean over all? (mean not yet implemented)

            Returns:
                {neuron: [results]} if neurons is a list, only [results] if only a single neuron is entered
        """
        ex = self.runner.experiments[parameter][repetition]

        if isinstance(neurons, list):
            results = {}
            for neuron in neurons:
                if isinstance(neuron, int):
                    neuron = C.NeuronOnHICANN(C.Enum(neuron))
                results[neuron] = [r[neuron][key] for r in ex.results]
        else:
            if isinstance(neurons, int):
                neuron = C.NeuronOnHICANN(C.Enum(neurons))
            results = [r[neuron][key] for r in ex.results]
        return results

    def plot_hist(self, parameter, key, step, repetition=0, histrange = 20, **kwargs):
        results = self.get_results(parameter, self.get_neurons(), key, repetition)
        results_list = np.array(results.values())[:,step]
        return plt.hist(results_list, bins=histrange, **kwargs)



