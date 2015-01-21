import cPickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pylogging
import pycake.calibrationrunner
import Coordinate as C
import pyhalbe
from bz2 import BZ2File
from gzip import GzipFile
from operator import itemgetter

class Reader(object):
    def __init__(self, runner, include_defects = True):
        """
        @param runner is either a CalibrationRunner instance or a CalibrationRunner pickled to a file (bz2 or gz)
        @param include_defects, if true, include neurons classified as defect
        """

        if isinstance(runner, pycake.calibrationrunner.CalibrationRunner):
            self.runner = runner
        else:
            if runner.endswith('.bz2'):
                f_open = BZ2File
            elif runner.endswith('.gz'):
                f_open = GzipFile
            else:
                f_open = open
            with f_open(os.path.expanduser(runner), 'rb') as infile:
                self.runner = cPickle.load(infile)

        self.include_defects = include_defects

    logger = pylogging.get("pycake.reader")

    def get_neurons(self):
        neurons = [nrn for nrn in self.runner.config.get_neurons()
                   if (self.include_defects == True or
                       self.runner.redman.hicann_with_backend.neurons().has(nrn))
               ]

        if not neurons:
            self.logger.warn("no neurons specified or all marked as defect")

        return neurons

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
        neurons = self.get_neurons()
        results = self.get_results(parameter, self.get_neurons(), key, repetition)
        results_list = np.array(results.values())[:,step]
        hist = plt.hist(results_list, label="{:.2f} +- {:.2f}, {}".format(np.mean(results_list)*1000, np.std(results_list)*1000, len(neurons)), **kwargs)
        config = self.runner.config.copy(parameter)
        step_valus = config.get_steps()[step]
        target_value = config.get_steps()[step]
        if draw_target_line and len(target_value) == 1:
            target_value = target_value.values()[0].value / 1000.0
            plt.axvline(target_value, linestyle='dashed', color='k', linewidth=1)
        return hist

    def plot_vs_neuron_number(self, parameter, key, step, repetition=0, draw_target_line=True, sort_by_shared_FG_block=False, **kwargs):
        neurons = self.get_neurons()
        results = self.get_results(parameter, self.get_neurons(), key, repetition)

        x_values = []
        results_list = []

        for n in neurons:
            if not sort_by_shared_FG_block:
                x_values.append(n.id().value())
            else:
                x_values.append(n.id().value()%256/2 + n.toSharedFGBlockOnHICANN().id().value()*128)

            results_list.append(results[n])

        p = plt.plot(*zip(*sorted(zip(x_values, results_list), key=itemgetter(0))))

        config = self.runner.config.copy(parameter)

        target_value = config.get_steps()[step]

        if draw_target_line and len(target_value) == 1:
            target_value = target_value.values()[0].value / 1000.0
            plt.axhline(target_value, linestyle='dashed', color='k', linewidth=1)

        return p

    def plot_vs_neuron_number_s(self, parameter, key, repetition=0, show_legend=False, **kwargs):

        nsteps = len(self.runner.config.copy(parameter).get_steps())

        fig = plt.figure()

        ps = [self.plot_vs_neuron_number(parameter, key, step, repetition, **kwargs) for step in xrange(nsteps)]

        if show_legend:
            #plt.legend()
            pass

        return fig, ps

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

    def plot_result(self, parameter, key, neurons=None, yfactor=1000, mark_top_bottom=True, average=False, **kwargs):

        fig = plt.figure()

        if neurons == None:
            neurons = self.get_neurons()

        coord_neurons = []
        for n in neurons:
            if isinstance(n, C.NeuronOnHICANN):
                coord_neurons.append(n)
            elif isinstance(n, int):
                coord_neurons.append(C.NeuronOnHICANN(C.Enum(n)))
            else:
                raise RuntimeError("unexpected type {} for neuron".format(type(n)))

        neurons = coord_neurons

        config = self.runner.config.copy(parameter)

        if mark_top_bottom:
            # http://stackoverflow.com/a/17210030/1350789
            colors = ['b','g']
            labels = ["top", "bottom"]
            [plt.plot(None,None,ls='-',marker='o',c=c,label=l) for c,l in zip(colors,labels)]
            plt.legend(labels)

        for vertical, color in zip([C.Y(0), C.Y(1)], ['b', 'g']):

            results = self.get_results(parameter, [n for n in neurons if n.y() == vertical], key, repetition=0)

            if results:

                xs = []
                ys = []

                for step, step_value in enumerate(config.get_steps()):

                    xs.append(step_value.values()[0].value)

                    if yfactor != 1:
                        ys_tmp = (np.array(results.values())*yfactor)[:,step]
                    else:
                        ys_tmp = (np.array(results.values()))[:,step]

                    if average:
                        ys.append(np.mean(ys_tmp))
                    else:
                        ys.append(ys_tmp)

                if mark_top_bottom:
                    plot = plt.plot(xs, ys, color=color, marker='o', **kwargs)
                else:
                    plot = plt.plot(xs, ys, **kwargs)

        return fig, plot
