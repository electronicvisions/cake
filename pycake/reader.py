import pickle
import os
import numpy as np
import pandas as pd
import pylogging
from pycake.calibrationrunner import CalibrationRunner
from pyhalco_common import Enum
import pyhalco_hicann_v2 as C
import pyhalbe
from operator import itemgetter
from .helpers.misc import load_pickled_file
from pycake.helpers.units import Second

class Reader(object):
    def __init__(self, runner, include_defects=True, backend_path=None):
        """
        @param runner is either a CalibrationRunner instance or a CalibrationRunner pickled to a file (bz2 or gz)
        @param include_defects, if true, include neurons classified as defect
        """

        if isinstance(runner, CalibrationRunner):
            self.runner = runner
        else:
            self.runner = CalibrationRunner.load(runner, backend_path=backend_path)
        self.include_defects = include_defects
        self.calibration_unit_cache = {}

        self.neurons_including_defects = self.runner.config.get_neurons()

        self.neurons_without_defects = [nrn for nrn in self.runner.config.get_neurons()
                                        if self.runner.redman.hicann_with_backend.neurons().has(nrn)]

        self.neurons_only_defect = [nrn for nrn in self.runner.config.get_neurons()
                                    if not self.runner.redman.hicann_with_backend.neurons().has(nrn)]

    logger = pylogging.get("pycake.reader")

    def get_neurons(self):

        neurons = self.neurons_including_defects if self.include_defects else self.neurons_without_defects

        if not neurons:
            self.logger.warn("no neurons specified or all marked as defect")

        return neurons

    def get_parameters(self,):
        return self.runner.to_run

    def get_calibration_unit(self, name, recurrence=0):
        """A calibration unit can occur several times in the configuration, for
        example if previous calibrations affect the precision. The second result for
        identical name can be obtained by recurrence=1 and so on.
        """

        key = (name, recurrence)
        try:
            return self.calibration_unit_cache[key]
        except KeyError:
            positions = self.runner.query_calibrations(name=name)
            if not positions:
                raise RuntimeError("No unit(s) found for name {}".format(name))
            unit = self.runner.get(name=name, pos=positions[recurrence])[0]
            self.calibration_unit_cache[key] = unit
            return unit

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
        unit = self.get_calibration_unit(name=parameter)
        ex = unit.experiment

        if isinstance(neuron, int):
            neuron = C.NeuronOnHICANN(Enum(neuron))
        df = ex.get_all_data([parameter, key]).sortlevel('neuron').loc[neuron]
        #FIXME: to be compatible with the plotting script, the df is converted
        #to a dict.
        df.index = df.index.droplevel(['shared_block', 'step'])
        return df[key].values

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
        unit = self.get_calibration_unit(name=parameter, recurrence=recurrence)
        ex = unit.experiment

        nrn_coord = []
        for neuron in neurons:
            if isinstance(neuron, int):
                neuron = C.NeuronOnHICANN(Enum(neuron))
            nrn_coord.append(neuron)
        # calibration name is I_gl_PSP, but parameter name still is I_gl -->
        # convert it
        if parameter is 'I_gl_PSP':
            parameter = 'I_gl'
        df = ex.get_all_data([parameter, key]).sortlevel('neuron').loc[nrn_coord]
        #FIXME: to be compatible with the plotting script, the df is converted
        #to a dict.
        if not isinstance(parameter, str):
            parameter = parameter.name
        nsteps = len(np.unique([d[2] for d in df.index.values]))
        df.index = df.index.droplevel(['shared_block', 'step'])
        results = {}
        for name, group in df.groupby(level='neuron'):
            if repetition is None:
                results.update({name : group[key].values})
            else:
                results.update({name : group[key].values[repetition*nsteps:(repetition+1)*nsteps]})
        return results

    def plot_trace(self, parameter, neuron, step, start=0, end=-1, recurrence=0):
        import matplotlib.pyplot as plt

        e = self.get_calibration_unit(name=parameter, recurrence=recurrence).experiment

        m = e.measurements[step]
        t = m.get_trace(neuron)

        if not t:
            raise KeyError("missing trace for {} {} {}".format(parameter, neuron, step))

        t_to_plot = [np.array(t[0][start:end])*1e6, t[1][start:end]]

        p = plt.plot(*t_to_plot)

        return p, t_to_plot

    def plot_hist(self, parameter, key, step, repetition=0, draw_target_line=True, **kwargs):
        import matplotlib.pyplot as plt
        neurons = self.get_neurons()
        results = self.get_results(parameter, neurons, key, repetition)
        results_list = np.array(list(results.values()))[:, step].astype(np.float)
        results_list = results_list[np.isfinite(results_list)]
        config = self.runner.config.copy(parameter)

        if not isinstance(list(config.get_steps()[0].values())[0], Second):
            hist_label = "{:.1f} +- {:.1f} mV, {}".format(np.mean(results_list)*1000, np.std(results_list)*1000, len(results_list))
        else:
            hist_label = "{:.2f} +- {:.2f} $\mu$s, {}".format(np.mean(results_list)*1e6, np.std(results_list)*1e6, len(results_list))

        hist = plt.hist(results_list, label=hist_label, **kwargs)
        step_valus = config.get_steps()[step]
        target_value = config.get_steps()[step]
        if draw_target_line and len(target_value) == 1:
            target_value = list(target_value.values())[0].value
            plt.axvline(target_value, linestyle='dashed', color='k', linewidth=1)
        return hist

    def plot_defect_neurons(self, sort_by_shared_FG_block=False, **kwargs):
        import matplotlib.pyplot as plt

        x_values = []
        results_list = []

        for n in self.neurons_only_defect:
            if not sort_by_shared_FG_block:
                x_values.append(n.toEnum().value())
            else:
                x_values.append(n.toEnum().value()%256/2 + n.toSharedFGBlockOnHICANN().toEnum().value()*128)

            results_list.append(1)

        fig = plt.figure()

        p = plt.plot(*list(zip(*sorted(zip(x_values, results_list), key=itemgetter(0)))), marker='x', linestyle="None",color='r')

        plt.gca().get_yaxis().set_ticks([])

        if not sort_by_shared_FG_block:
            plt.xlabel("Neuron")
        else:
            plt.xlabel("Shared FG block*128 + Neuron%256/2")

        plt.ylabel("Defect")
        plt.xlim(0, 512)
        plt.ylim(0, 2)

        return fig, p

    def plot_vs_neuron_number(self, parameter, key, step, repetition=0, draw_target_line=True, sort_by_shared_FG_block=False, **kwargs):
        import matplotlib.pyplot as plt
        neurons = self.get_neurons()
        results = self.get_results(parameter, self.get_neurons(), key, repetition)

        x_values = []
        results_list = []

        for n in neurons:
            if not sort_by_shared_FG_block:
                x_values.append(n.toEnum().value())
            else:
                x_values.append(n.toEnum().value()%256/2 + n.toSharedFGBlockOnHICANN().toEnum().value()*128)

            results_list.append(results[n])

        p = plt.plot(*list(zip(*sorted(zip(x_values, results_list), key=itemgetter(0)))), marker='x', linestyle="None")

        config = self.runner.config.copy(parameter)

        target_value = config.get_steps()[step]

        if draw_target_line and len(target_value) == 1:
            target_value = list(target_value.values())[0].value
            plt.axhline(target_value, linestyle='dashed', color='k', linewidth=1)

        return p

    def plot_vs_neuron_number_s(self, parameter, key, repetition=0, show_legend=False, **kwargs):
        import matplotlib.pyplot as plt

        nsteps = len(self.runner.config.copy(parameter).get_steps())

        fig = plt.figure()

        ps = [self.plot_vs_neuron_number(parameter, key, step, repetition, **kwargs) for step in range(nsteps)]

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

        hists = [self.plot_hist(parameter, key, step, repetition, **kwargs) for step in range(nsteps)]

        if show_legend:
            plt.legend()

        return fig, hists

    def plot_std(self, parameter, hicann_parameter, key, step, **kwargs):
        import matplotlib.pyplot as plt

        if not isinstance(hicann_parameter, str):
            hicann_parameter = hicann_parameter.name
        unit = self.runner.get_single(name=parameter)
        e = unit.experiment
        data = e.get_all_data([hicann_parameter, key])
        idx_len = len(df[hicann_parameter].unique())
        std = data.set_index(hicann_parameter, append=True).groupby(level=['neuron', hicann_parameter]).std()
        #create step index for compatibility with function call
        std_values = std.swaplevel(0, 'step').loc[step].values

        hist = plt.hist(std_values, **kwargs)
        return hist

    def plot_result(self, parameter, key, neurons=None, yfactor=1, mark_top_bottom=True, average=False, marker='o', step_key=None, **kwargs):
        import matplotlib.pyplot as plt

        fig = plt.figure()

        if neurons == None:
            neurons = self.get_neurons()

        coord_neurons = []
        for n in neurons:
            if isinstance(n, C.NeuronOnHICANN):
                coord_neurons.append(n)
            elif isinstance(n, int):
                coord_neurons.append(C.NeuronOnHICANN(Enum(n)))
            else:
                raise RuntimeError("unexpected type {} for neuron".format(type(n)))

        neurons = coord_neurons

        config = self.runner.config.copy(parameter)

        if mark_top_bottom:
            # http://stackoverflow.com/a/17210030/1350789
            colors = ['b','g']
            labels = ["top", "bottom"]
            [plt.plot([],[],ls='-',marker=marker,c=c,label=l) for c,l in zip(colors,labels)]
            plt.legend(labels, loc=2)

        for vertical, color in zip([C.Y(0), C.Y(1)], ['b', 'g']):

            results = self.get_results(parameter, [n for n in neurons if n.y() == vertical], key, repetition=0)

            if results:

                xs = []
                ys = []
                yerrs = []

                for step, step_value in enumerate(config.get_steps()):

                    if step_key is None:
                        val = list(step_value.values())[0]
                    else:
                        val = step_value[step_key]

                    if not isinstance(val, Second):
                        xs.append(val.toDAC().value)
                    else:
                        xs.append(val.value)

                    if yfactor != 1:
                        ys_tmp = (np.array(list(results.values()))*yfactor)[:,step]
                    else:
                        ys_tmp = (np.array(list(results.values())))[:,step]

                    if average:
                        ys.append(np.mean(ys_tmp))
                        yerrs.append(np.std(ys_tmp))
                    else:
                        ys.append(ys_tmp)

                if mark_top_bottom:
                    if average:
                        plot = plt.errorbar(xs, ys, yerr=yerrs, color=color, marker=marker, **kwargs)
                    else:
                        plot = plt.plot(xs, ys, color=color, marker=marker, **kwargs)
                else:
                    if average:
                        plot = plt.errorbar(xs, ys, yerr=yerrs, **kwargs)
                    else:
                        plot = plt.plot(xs, ys, **kwargs)

        return fig, plot
