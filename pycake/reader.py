import cPickle
import os
import numpy as np
import pylogging
from pycake.calibrationrunner import CalibrationRunner
import Coordinate as C
import pyhalbe
from operator import itemgetter
from helpers.misc import load_pickled_file

class Reader(object):
    def __init__(self, runner, include_defects = True):
        """
        @param runner is either a CalibrationRunner instance or a CalibrationRunner pickled to a file (bz2 or gz)
        @param include_defects, if true, include neurons classified as defect
        """

        if isinstance(runner, CalibrationRunner):
            self.runner = runner
        else:
            self.runner = CalibrationRunner.load(runner)
        self.include_defects = include_defects
        self.calibration_unit_cache = {}

        self.neurons_including_defects = self.runner.config.get_neurons()

        self.neurons_without_defects = [nrn for nrn in self.runner.config.get_neurons()
                                        if self.runner.redman.hicann_with_backend.neurons().has(nrn)]

    logger = pylogging.get("pycake.reader")

    def get_neurons(self):

        neurons = self.neurons_including_defects if self.include_defects else self.neurons_without_defects

        if not neurons:
            self.logger.warn("no neurons specified or all marked as defect")

        return neurons

    def get_parameters(self,):
        return self.runner.to_run

    def get_calibration_unit(self, name, recurrence=0):

        key = (name, recurrence)
        try:
            return self.calibration_unit_cache[key]
        except KeyError:
            result = self.runner.get(name=name, pos=self.runner.query_calibrations(name=name)[recurrence])[0]
            self.calibration_unit_cache[key] = result
            return result

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
        step = self.get_calibration_unit(name=parameter)
        ex = step.experiment

        if isinstance(neuron, int):
            neuron = C.NeuronOnHICANN(C.Enum(neuron))
        return [r[neuron].get(key, None) for r in ex.results]

    def get_results(self, parameter, neurons, key, repetition=None, recurrence=0):
        """ Get measurement results for one neuron.

            Args:
                parameter: which parameter?
                neurons: list of neuron coordinates or ids
                key: which key? (e.g. 'mean', 'std', 'baseline', ...)
                repetition: if not None only values for given repetition are returned
                recurrence: recurrence of calibration unit

            Returns:
                {neuron: [results]} if neurons
        """

        step = self.get_calibration_unit(name=parameter, recurrence=recurrence)

        ex = step.experiment
        nsteps = len(self.runner.config.copy(parameter).get_steps())

        results = {}
        for neuron in neurons:
            if isinstance(neuron, int):
                neuron = C.NeuronOnHICANN(C.Enum(neuron))

            results[neuron] = np.array([r[neuron].get(key, None) for r in ex.results])
            if repetition != None:
                if np.ndim(results[neuron]) > 1:
                    raise RuntimeError("retrieval of results is not implemented for multi-dimensional sweeps")
                else:
                    results[neuron] = list(results[neuron][[nsteps*repetition + step for step in range(nsteps)]])

        return results

    def plot_hist(self, parameter, key, step, repetition=0, draw_target_line=True, **kwargs):
        import matplotlib.pyplot as plt
        neurons = self.get_neurons()
        results = self.get_results(parameter, neurons, key, repetition)
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
        import matplotlib.pyplot as plt
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
        import matplotlib.pyplot as plt

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
        import matplotlib.pyplot as plt

        nsteps = len(self.runner.config.copy(parameter).get_steps())

        fig = plt.figure()

        hists = [self.plot_hist(parameter, key, step, repetition, **kwargs) for step in xrange(nsteps)]

        if show_legend:
            plt.legend()

        return fig, hists

    def plot_std(self, parameter, hicann_parameter, key, step, **kwargs):
        import matplotlib.pyplot as plt

        step = self.runner.get_single(name=parameter)
        e = step.experiment

        hist = plt.hist([e.get_mean_results(n,hicann_parameter,[key])[0][step][-1] for n in self.get_neurons()], **kwargs)

        return hist

    def plot_result(self, parameter, key, neurons=None, yfactor=1000, mark_top_bottom=True, average=False, **kwargs):
        import matplotlib.pyplot as plt

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
                yerrs = []

                for step, step_value in enumerate(config.get_steps()):

                    xs.append(step_value.values()[0].value)

                    if yfactor != 1:
                        ys_tmp = (np.array(results.values())*yfactor)[:,step]
                    else:
                        ys_tmp = (np.array(results.values()))[:,step]

                    if average:
                        ys.append(np.mean(ys_tmp))
                        yerrs.append(np.std(ys_tmp))
                    else:
                        ys.append(ys_tmp)

                if mark_top_bottom:
                    if average:
                        plot = plt.errorbar(xs, ys, yerr=yerrs, color=color, marker='o', **kwargs)
                    else:
                        plot = plt.plot(xs, ys, color=color, marker='o', **kwargs)
                else:
                    if average:
                        plot = plt.errorbar(xs, ys, yerr=yerrs, **kwargs)
                    else:
                        plot = plt.plot(xs, ys, **kwargs)

        return fig, plot
