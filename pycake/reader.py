import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pycake.calibrationrunner
import Coordinate as C
import pyhalbe
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

    def get_parameters(self):
        return self.runner.experiments.keys()

    def get_result(self, parameter, neuron, key):
        """ Get measurement results for one neuron.

            Args:
                parameter: which parameter?
                neurons: neuron coordinate or id
                key: which key? (e.g. 'mean', 'std', 'baseline', ...)
                repetition: single repetition or mean over all? (mean not yet implemented)

            Returns:
                list of all results
        """
        ex = self.runner.experiments[parameter]

        if isinstance(neuron, int):
            neuron = C.NeuronOnHICANN(C.Enum(neuron))
        return [r[neuron].get(key, None) for r in ex.results]

    def get_results(self, parameter, neurons, key, repetition=None):
        """ Get measurement results for one neuron.

            Args:
                parameter: which parameter?
                neurons: list of neuron coordinates or ids
                key: which key? (e.g. 'mean', 'std', 'baseline', ...)
                repetition: if not None only values for given repetition are returned

            Returns:
                {neuron: [results]} if neurons
        """
        ex = self.runner.experiments[parameter]
        nsteps = len(self.runner.config.copy(parameter).get_steps())

        results = {}
        for neuron in neurons:
            if isinstance(neuron, int):
                neuron = C.NeuronOnHICANN(C.Enum(neuron))

            results[neuron] = [r[neuron].get(key, None) for r in ex.results]

            if repetition != None:
                results[neuron] = list(np.array(results[neuron])[:,[nsteps*repetition + step for step in range(nsteps)]])
                #                                                   ^^^^^^^^^^^^^^^^^^^^^^^^
                # FIXME: the indexing above is wrong for multi-dimensional parameter scans

        return results

    def plot_hist(self, parameter, key, step, repetition=0, draw_target_line=True, **kwargs):
        results = self.get_results(parameter, self.get_neurons(), key, repetition)
        results_list = np.array(results.values())[:,step]
        hist = plt.hist(results_list, label="{:.2f} +- {:.2f}".format(np.mean(results_list)*1000, np.std(results_list)*1000), **kwargs)
        config = self.runner.config.copy(parameter)
        step_valus = config.get_steps()[step]
        target_value = config.get_steps()[step]
        if draw_target_line and len(target_value) == 1:
            target_value = target_value.values()[0].value / 1000.0
            plt.axvline(target_value, linestyle='dashed', color='k', linewidth=1)
        return hist

    def plot_hists(self, parameter, key, repetition=0, show_legend=False, **kwargs):
        """ Returns figure and histograms for all steps
        """

        nsteps = len(self.runner.config.copy(parameter).get_steps())

        fig = plt.figure()

        hists = [self.plot_hist(parameter, key, step, repetition, **kwargs) for step in xrange(nsteps)]

        if show_legend:
            plt.legend()

        return fig, hists

    def plot_std(self, parameter, hicann_parameter, key, step, **kwargs):

        e = self.runner.experiments[parameter]

        hist = plt.hist([e.get_mean_results(n,hicann_parameter,[key])[0][step][-1] for n in self.get_neurons()], **kwargs)

        return hist

    def plot_result(self, parameter, key, **kwargs):

        results = self.get_results(parameter, self.get_neurons(), key, repetition=0)

        config = self.runner.config.copy(parameter)

        xs = []
        ys = []

        for step, step_value in enumerate(config.get_steps()):

            xs.append(step_value.values()[0].value)
            ys.append(np.array(results.values())[:,step])

        plot = plt.plot(xs, ys, **kwargs)

        return plot

    def outliers(self, parameter, key, min_value, max_value, neurons=range(512)):

        n_above_max = sum([v[0] > max_value for v in self.get_results(parameter, neurons, key).values()])
        n_below_min = sum([v[0] < min_value for v in self.get_results(parameter, neurons, key).values()])

        return n_above_max, n_below_min
