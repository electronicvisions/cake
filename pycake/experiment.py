import pylogging
import time
import numpy
import copy
import os
import pandas
import itertools

from Coordinate import iter_all
from Coordinate import NeuronOnHICANN
from pyhalbe.HICANN import neuron_parameter, shared_parameter

plogger = pylogging.get('progress.experiment')

class Experiment(object):
    """Base class for running experiments

    Attributes:
        analyzer: list of analyzer objects (or just one)
        measurements: list of successful exececuted measurements
    """

    logger = pylogging.get("pycake.experiment")

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.measurements = []
        self.results = []
        self.fg_values = {}
        self.run_time = 0.0

    def run(self):
        """Run the experiment and process results."""
        return list(self.iter_measurements())

    def finished(self):
        """Returns true after all measurements have been done"""
        raise NotImplementedError("Not implemented in {}".format(
            type(self).__name__))

    def iter_measurements(self):
        cls_name = type(self).__name__
        raise NotImplementedError("Not implemented in {}".format(cls_name))

    def save_traces(self):
        cls_name = type(self).__name__
        raise NotImplementedError("Not implemented in {}".format(cls_name))

    def set_read_fg_values(self, parameters):
        """
        Read back the fg_values for the given parameters. This is only done
        for the half of the parameters reachable by the active analog channel
        """
        self.fg_values = {p: {} for p in parameters}

    def get_measurement(self, i):
        try:
            return self.measurements[i]
        except IndexError:
            return None

    def get_result_keys(self):
        """
        Iterates over all results and collects all available keys

        Return:
            list: Containing all keys
        """
        results = set()
        for step_results in self.results:
            for result in step_results.itervalues():
                results |= set(result.keys())
        return sorted(results)

    def get_parameter_names(self):
        """
        Iterates over all measurments (including incompleted) and returns a
        list of names of all parameter passed to the measurements.

        Return:
            list: Containing all neuron, shared and step parameters
        """
        parameters = neuron_parameter.values.values()
        parameters += shared_parameter.values.values()
        # Remove non-parameter enum entries, note getattr is needed because
        # of python name mangling used for private members
        parameters.remove(getattr(neuron_parameter, '__last_neuron'))
        parameters.remove(getattr(shared_parameter, '__last_shared'))

        step_parameters = set()
        for measurment in self.measurements_to_run:
            step_parameters |= set(
                p for p in measurment.step_parameters
                if not isinstance(p, (neuron_parameter, shared_parameter)))
        return parameters + sorted(step_parameters)

    def append_measurement_and_result(self, measurement, result):
        self.measurements.append(measurement)
        self.results.append(result)

    def clear_measurements_and_results(self):
        self.measurements[:] = []
        self.results[:] = []

    def get_parameters_and_results(self, neuron, parameters, result_keys):
        """ Read out parameters and result for the given neuron.
        Shared parameters are read from the corresponding FGBlock.
        If any of the given keys could not be obtained from a measurement,
        this measurement is ignored.

        Args:
            neuron: parameters matching this neuron
            parameter: [list] parameters to read (neuron_parameter or
                       shared_parameter)
            result_keys: [list] result keys to append

        Return:
            2D numpy array: each row contains the containing parameters
                in the requested order, followed by the requested keys

        Raises:
            ValueError, if no results are found for the given neuron.
        """
        values = []
        for measurement, result in itertools.izip_longest(
                self.measurements_to_run, self.results, fillvalue={}):
            value = measurement.get_parameters(neuron, parameters)
            try:
                nrn_results = result[neuron]
                value.extend(nrn_results[key] for key in result_keys)
            except KeyError:
                continue
            values.append(value)

        if not values:
            raise ValueError("No results for {}".format(neuron))

        try:
            return numpy.array(values)
        except ValueError:
            return values

    def get_parameters(self, neuron, parameters):
        """short for get_parameters_and_results(neuron, parameters, tuple())"""
        return self.get_parameters_and_results(neuron, parameters, tuple())

    def get_results(self, neuron, keys):
        """short for get_parameters_and_results(neuron, tuple(), keys)"""
        return self.get_parameters_and_results(neuron, tuple(), keys)

    def get_data(self, neuron, parameters, result_keys):
        """ Read out parameters and result for the given neuron.
        Shared parameters are read from the corresponding FGBlock.
        If any of the given keys could not be obtained from a measurement,
        NaN is returned. If no result at all could be found, a ValueError is
        raised.

        Args:
            neuron: parameters matching this neuron
            parameter: [list] parameters to read (neuron_parameter,
                       shared_parameter or string). Floating gate parameters
                       will be read from the floating gate configuration.
                       Sting parameters will be returned from step_parameters.
        Return:
            pandas.DataFrame: Each requested parameter/result is one column,
                              indexed with the measurement step number.

        Raises:
            ValueError, if no results are found for the given neuron.
        """
        names = [p if isinstance(p, basestring) else p.name
                 for p in parameters]
        names.extend(result_keys)
        result_keys = list(result_keys)
        values = []
        for msr, result in itertools.izip_longest(
                self.measurements_to_run, self.results, fillvalue={}):
            row = msr.get_parameters(neuron, parameters)
            nrn_results = result.get(neuron, {})
            row.extend(nrn_results.get(key, numpy.nan) for key in result_keys)
            values.append(row)
        values = pandas.DataFrame(values, columns=names)
        if result_keys and values[result_keys].isnull().values.all():
            raise ValueError("Could not find any requested results")
        return values

    def get_all_data(self, parameters=None, result_keys=None, numeric_index=False):
        """ Read out parameters and result for all neurons.
        The results of get_data are concatenated for all neurons. Neurons
        without any results are ignored.

        Args:
            neuron: parameters matching this neuron
            parameter: [list] parameters to read (neuron_parameter or
                       shared_parameter), if None all parameters are returned
            result_keys: [list] result keys to append, if None all keys are
                         returned
            numeric_index: [bool] don't use NeuronOnHICANN as index, but ints,
                           this allows to save/load the data with plain pandas

        Return:
            pandas.DataFrame: Each requested parameter/result is one column.
                              The DataFrame is indexed with a multi index, the
                              first level is the NeuronOnHICANN coordinate, the
                              second level the measurement step.
        """
        if parameters is None:
            parameters = self.get_parameter_names()
        if result_keys is None:
            result_keys = self.get_result_keys()
        else:
            result_keys = list(result_keys)

        data = {}
        for nrn in iter_all(NeuronOnHICANN):
            try:
                idx = nrn, nrn.toSharedFGBlockOnHICANN()
                if numeric_index:
                    idx = tuple(int(v.id()) for v in idx)
                data[idx] = self.get_data(nrn, parameters, result_keys)
            except ValueError:
                pass
        return pandas.concat(
            data, names=['neuron', 'shared block', 'step'])

    def get_initial_data(self, numeric_index=False):
        """
        Colltect the initial data of the expriment in pandas data structures
        Args:
            numeric_index: [bool] don't use NeuronOnHICANN as index, but ints,
                           this allows to save/load the data with plain pandas

        Returns:
            (pandas.DataFrame, pandas.Series): Neuron specific values are
            collected in data the DataFrame with the neuron coordinate as
            index. All other keys are returned as Series.
        """
        def nrn_id(n):
            return int(n.id().value()) if numeric_index else n

        def value(v):
            if v is None:
                return numpy.nan
            else:
                return v

        nrn_data = {nrn_id(k): value(v)
                    for k, v in self.initial_data.iteritems()
                    if isinstance(k, NeuronOnHICANN)}
        nrn_data = pandas.DataFrame.from_dict(nrn_data).T
        nrn_data.index.names = ['neuron']

        data = pandas.Series({
            k: value(v)
            for k, v in self.initial_data.iteritems()
            if not isinstance(k, NeuronOnHICANN)})

        return data, nrn_data

    def get_mean_results(self, neuron, parameter, result_keys):
        """ Read out parameters and result for the given neuron.
        Shared parameters are read from the corresponding FGBlock.

            Args:
                neuron: parameters matching this neuron
                parameter: parameter to read (neuron_parameter or
                           shared_parameter)
                result_keys: [list] result keys to append
            Return:
                [2D numpy array]:
                each row contains the containing parameter value, count, mean
                and std of the requested key.
        """
        data = self.get_parameters_and_results(
                neuron, (parameter, ), result_keys)
        values = numpy.unique(data[:, 0])
        results = [numpy.empty((len(values), 4)) for _ in result_keys]
        key_indices = numpy.arange(len(result_keys)) + 1

        for row, value in enumerate(values):
            subset = data[data[:, 0] == value]
            count = len(subset)
            mean = numpy.mean(subset, axis=0)
            std = numpy.std(subset, axis=0)
            for ii in key_indices:
                results[ii-1][row] = (mean[0], count, mean[ii], std[ii])
        return results

    def print_parameters(self, neuron, parameters):
        """
        Prints the given parameters in a nice ASCII table.

        Arguments:
            parameters: iterable containing the parameters
        Returns:
            [str] parameters written in table form
        """
        t_field = "{{:>{w}}}"

        data = self.get_parameters_and_results(neuron, parameters, tuple())
        fields = [len(p.name) for p in parameters]
        header = "|     | " + " | ".join(p.name for p in parameters) + " |"
        lines = [header, len(header) * '-']
        for ii, row in enumerate(data):
            f = [t_field.format(w=3).format(ii)]
            f.extend(t_field.format(w=w).format(v) for w, v in zip(fields, row))
            lines.append("| " + " | ".join(f) + " |")
        return "\n".join(lines)

    def get_fg_values(self, numeric_index=False):
        """
        Return analog read floating gate values

        Args:
            numeric_index: [bool] don't use NeuronOnHICANN as index, but
                           integers, this allows to use the data without HALbe

        Returns:
            pandas.DataFrame with MultiIndex 'step', 'block', 'neuron', the
            columns are indexed by parameter name
        """
        fg_values = {}
        for p in self.fg_values.keys():
            values = pandas.concat(self.fg_values[p],
                                  names=['step', 'shared block', 'neuron'])
            values = values.reorder_levels(['neuron', 'shared block', 'step'])
            if numeric_index:
                values.index = pandas.MultiIndex.from_tuples(
                    [(nrn.id().value(), blk.id().value(), step)
                     for nrn, blk, step in values.index.values],
                    names=values.index.names)
            fg_values[p.name] = values
        if fg_values:
            return pandas.concat(fg_values, axis='columns')
        else:
            return pandas.DataFrame()


class SequentialExperiment(Experiment):
    """ Takes a list of measurements and analyzers.
        Then, it runs the measurements and analyzes them.
        Traces can be saved to hard drive or discarded after analysis.

        Experiments can be continued after a crash by just loading and starting them again.

        Args:
            measurements: list of Measurement objects
            analyzer: list of analyzer objects (or just one)
            save_traces: (True|False) should the traces be saved to HDD
    """
    def __init__(self, measurements, analyzer, repetitions):
        Experiment.__init__(self, analyzer)
        self.measurements_to_run = measurements
        self.repetitions = repetitions
        self.initial_data = {}
        self.initial_measurements = []

    def finished(self):
        """Returns true after all measurements have been done"""
        return len(self.measurements_to_run) == len(self.measurements)

    def add_initial_measurement(self, measurement, analyzer):
        self.initial_measurements.append((measurement, analyzer))

    def run_initial_measurements(self):
        """Run preparation measurements."""

        if self.initial_measurements:
            self.logger.INFO("Running initial measurements.")
            plogger.debug("Running initial measurements.")
        for measurement, analyzer in self.initial_measurements:
            if measurement.done:
                self.logger.INFO("Initial measurement already done. "
                                 "Going on with next one.")
                continue
            else:
                t_start = time.time()
                result = measurement.run_measurement(
                    analyzer, self.initial_data)
                self.initial_data.update(result)
                self.run_time += time.time() - t_start

    def save_traces(self, path):
        for mid, measurement in enumerate(self.measurements_to_run):
            filename = os.path.join(path, "{}.hdf5".format(mid))
            measurement.save_traces(filename)
        for mid, (measurement, _) in enumerate(self.initial_measurements):
            filename = os.path.join(path, "intial_{}.hdf5".format(mid))
            measurement.save_traces(filename)


    def iter_measurements(self):
        self.run_initial_measurements()
        i_max = len(self.measurements_to_run)
        for i, measurement in enumerate(self.measurements_to_run):
            if not measurement.done:
                t_start = time.time()
                self.logger.INFO("Running measurement {}/{}".format(i+1, i_max))
                plogger.debug("Running measurement {}/{}".format(i+1, i_max))
                result = measurement.run_measurement(self.analyzer, self.initial_data)
                self.append_measurement_and_result(measurement, result)
                self.run_time += time.time() - t_start
                yield True # Used to save state of runner 
            else:
                self.logger.INFO("Measurement {}/{} already done. Going on with next one.".format(i+1, i_max))
                yield False
        return

    # Compatibility for old pickels
    def __setstate__(self, state):
        if 'measurements_to_run' not in state:
            state['measurements_to_run'] = copy.copy(state['measurements'])
        if 'initial_data' not in state:
            state['initial_data'] = {}
        if 'initial_measurements' not in state:
            state['initial_measurements'] = []
        self.__dict__.update(state)


class I_pl_Experiment(SequentialExperiment):
    """ I_pl experiment uses post-processing to calculate tau_ref results.
        Otherwise, it does the same as SequentialExperiment
    """

    def prepare_x(self, x, amplitudes, mean_reset_times):
        """ Prepares x values for fit
            Here, the refractory period needs to be calculated from dt of spikes.
            tau_ref is calculated via ISI_n - ISI_0, where ISI_0 is the mean ISI of
            all measurements with I_pl = 1023 DAC. Correction for amplitude variations
            are also done, resulting the final calculation of tau_ref for step n:
                tau_ref_n = ISI_n - amplitude_n / amplitude_0 * ISI_0
        """
        x = numpy.array(x)
        amplitudes = numpy.array(amplitudes)
        mean_reset_times = numpy.array(mean_reset_times)
        tau0 = mean_reset_times[-1]
        # Correct ISI0 for spike amplitude differences
        corrected_ISI0 = x[-1] * amplitudes/amplitudes[-1]
        #tau_refracs = x - ISI0 + tau0
        tau_refracs = x - corrected_ISI0 + tau0
        return tau_refracs

# Compatibility for old pickels
BaseExperiment = SequentialExperiment


class IncrementalExperiment(SequentialExperiment):

    def __init__(self, measurements, analyzer, configurator, configurator_args):
        SequentialExperiment.__init__(self, measurements, analyzer, repetitions=1)
        self.configurator = configurator
        self.configurator_args = configurator_args

    def iter_measurements(self):
        if self.finished():
            return

        self.clear_measurements_and_results()

        self.run_initial_measurements()
        t_start = time.time()
        sthal = copy.deepcopy(self.measurements_to_run[0].sthal)
        sthal.write_config()
        configurator = self.configurator(**self.configurator_args)
        i_max = len(self.measurements_to_run)
        for i, measurement in enumerate(self.measurements_to_run):
            plogger.debug("Running measurement {}/{}".format(i+1, i_max))
            sthal.hicann.copy(measurement.sthal.hicann)
            measurement.sthal = sthal
            result = measurement.run_measurement(
                self.analyzer, additional_data=self.initial_data,
                configurator=configurator, disconnect=False)
            for parameter, values in self.fg_values.iteritems():
                values[i] = sthal.read_floating_gates(parameter)
            measurement.sthal = copy.deepcopy(measurement.sthal)
            self.append_measurement_and_result(measurement, result)
            yield False # No need to save, because we can't resume any way
        sthal.disconnect()
        self.run_time += time.time() - t_start
        yield True
