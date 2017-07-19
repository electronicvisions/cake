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
        results_index = pandas.MultiIndex(
                levels=[[],[],[]], labels=[[],[],[]],
                names=['neuron', 'shared_block', 'step'])
        self.results = pandas.DataFrame(index=results_index)
        self.fg_values = {}
        self.run_time = 0.0

    # Compatibility for old pickels
    def __setstate__(self, state):
        if isinstance(state['results'], list):
            data = zip(state['measurements'], state['results'], itertools.count())
            state['results'] = pandas.concat([
                self.convert_to_dataframe(measurement, result, step)
                for measurement, result, step in data])
        self.__dict__.update(state)

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
        step = len(self.measurements)
        self.measurements.append(measurement)
        self.results = pandas.concat([self.results,
            self.convert_to_dataframe(measurement, result, step)])

    def clear_measurements_and_results(self):
        self.measurements[:] = []
        self.results = pandas.DataFrame()

    @staticmethod
    def convert_to_dataframe(measurement, result, step):
        """
        collects all parameters of 'measurement' and all results from 'result' to
        create a DataFrame.

        Args:
            measurement: contains the parameters of the measurement
            result: measured values for the given measurement
            numeric_index: If the NeuronOnHICANN(Enum(i)) should be converted
                           to i (similarly for FGBlockOnHICANN).

        Returns:
            pandas.DataFrame: index: (NeuronOnHICANN, FGBlockOnHICANN),
                              columns: all parameters and measurement results
        """
        parameters = neuron_parameter.values.values()
        parameters += shared_parameter.values.values()
        # Remove non-parameter enum entries, note getattr is needed because
        # of python name mangling used for private members
        parameters.remove(getattr(neuron_parameter, '__last_neuron'))
        parameters.remove(getattr(shared_parameter, '__last_shared'))
        step_parameters = set(
            p for p in measurement.step_parameters
            if not isinstance(p, (neuron_parameter, shared_parameter)))
        parameters = parameters + sorted(step_parameters)
        names = [p if isinstance(p, basestring) else p.name
                 for p in parameters]

        data = {}
        for nrn in iter_all(NeuronOnHICANN):
            try:
                idx = nrn, nrn.toSharedFGBlockOnHICANN()
                parameter_values = measurement.get_parameters(nrn, parameters)
                params_dict = {name : parameter_values[i] for i,name in enumerate(names)}
                nrn_results = result.get(nrn, {})
                nrn_results.update(params_dict)
                data[idx] = nrn_results
            except ValueError:
                pass
        results_df = pandas.DataFrame.from_dict(data, orient='index')
        results_df.index.names = ['neuron', 'shared_block']
        results_df['step'] = step
        results_df.set_index('step', append=True, inplace=True)
        # The parameters 'V_clra', 'V_clrc', 'V_bout', 'V_bexp' only exist in
        # one half of the shared blocks (e.g. V_clra : (0,2) or V_clrc : (1,3))
        # but are shared with the other shared_blocks. Thus, these values are
        # manually added to the blocks which would contain NaN otherwise
        for param in ['V_clra', 'V_clrc', 'V_bout', 'V_bexp']:
            val = results_df[param][~pandas.isnull(results_df[param])].values[0]
            results_df[param] = val
        return results_df

    def get_all_data(self, keys=None, numeric_index=False):
        """ Read out parameters and result for all neurons.
            Only if parameters and result_keys are None, values for all parameters and result_keys are returned.

        Args:
            keys: [list] keys to read (neuron_parameter, shared_parameter, or a
                key from the measurement results). If None all data is returned
            numeric_index: [bool] don't use NeuronOnHICANN as index, but ints,
                           this allows to save/load the data with plain pandas

        Returns:
            pandas.DataFrame: Each requested key is one column.
                              The DataFrame is indexed with a multi index, the
                              first level is the NeuronOnHICANN coordinate, the
                              second level the FGBlock coordinate.
        """
        results = self.results
        if numeric_index:
            results = results.copy()
            idx = [(index[0].id().value(), index[1].id().value(), index[2])
                     for index in results.index]
            idx = pandas.MultiIndex.from_tuples(idx, names=['neuron', 'shared_block', 'step'])
            results.index = idx

        if keys is None:
            return results
        else:
            assert isinstance(keys, list)
            names = [str(p) for p in keys]
            return results.loc[:, names]

    def get_parameters(self, numeric_index=False):
        """
        Get all parameter values.

        Args:
            numeric_index: [bool] don't use NeuronOnHICANN as index, but ints,
                           this allows to save/load the data with plain pandas

        Returns:
            pandas.DataFrame: Each parameter is one column.
                              The DataFrame is indexed with a multi index, the
                              first level is the NeuronOnHICANN coordinate, the
                              second level the FGBlock coordinate.
        """
        parameters = self.get_parameter_names()
        names = [p if isinstance(p, basestring) else p.name
                 for p in parameters]
        return self.get_all_data(keys=names, numeric_index=numeric_index)

    def get_results(self, numeric_index=False):
        """
        Get all result values.

        Args:
            numeric_index: [bool] don't use NeuronOnHICANN as index, but ints,
                           this allows to save/load the data with plain pandas

        Returns:
            pandas.DataFrame: Each result is one column.
                              The DataFrame is indexed with a multi index, the
                              first level is the NeuronOnHICANN coordinate, the
                              second level the FGBlock coordinate.
        """
        parameters = self.get_parameter_names()
        names = [p if isinstance(p, basestring) else p.name
                 for p in parameters]
        keys = list(self.results.columns)
        for name in names:
            keys.remove(name)
        return self.get_all_data(keys=keys, numeric_index=numeric_index)

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

    def get_mean_results(self, parameter, result_keys):
        """
        Read out parameters and result for the given neuron.
        Shared parameters are read from the corresponding FGBlock.

            Args:
                parameter: [parameter] (neuron_parameter or shared_parameter).
                    Determines the level of averaging (i.e., the mean is
                    taken over the same parameter values)
                result_keys: [list] keys to average over

            Return:
                [pandas.DataFrame] Contains count, mean, std, min, 25%, 50%,
                75%, max (default of pandas.DataFrame.describe(). If parameter
                is a neuron parameter, returns data for each neuron, otherwise
                returns data for each FGBlock
        """
        if isinstance(parameter, shared_parameter):
            group_level = 'shared_block'
        else:
            group_level = 'neuron'
        data = self.get_all_data([parameter, result_keys])
        mean = data.set_index(parameter, append=True).groupby(
               level=[group_level, parameter_name]).describe()

        return mean

    def print_parameters(self, neuron, parameters):
        """
        TODO: Replace by DataFrame.to_string?
        Prints the given parameters in a nice ASCII table.

        Arguments:
            parameters: iterable containing the parameters
        Returns:
            [str] parameters written in table form
        """
        t_field = "{{:>{w}}}"

        data = self.get_all_data(keys=parameters).sortlevel('neuron').loc[neuron]
        fields = [len(p.name) for p in parameters]
        header = "|     | " + " | ".join(p.name for p in parameters) + " |"
        lines = [header, len(header) * '-']
        for ii in range(len(data)):
            f = [t_field.format(w=3).format(ii)]
            f.extend(t_field.format(w=w).format(v) for w, v in zip(fields, data.iloc[ii]))
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
        Experiment.__setstate__(self, state)


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

    def run_initial_measurements(self):
        """Run preparation measurements.
           The measurements are averaged over the number of repetitions.
           The standard error is calculated from the std deviation of
           the single measurements.
        """

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
                values_all = []

                #get measurement results
                for i in range(self.repetitions):
                    result = measurement.run_measurement(
                        analyzer, self.initial_data)
                    values = [neuron_values.values() for neuron_values in result.itervalues()]
                    values_all.append(values)
                value_keys = result.values()[0].keys()
                neuron_coords = result.keys()
                values_all = numpy.array(values_all)

                #get index of the standard deviation to find it in the values_all array
                std_isi_idx = [i for i,j in enumerate(value_keys) if j == 'std_isi'][0]
                #calculate variance from std deviation
                values_all[:,:,std_isi_idx] = values_all[:,:,std_isi_idx]**2
                #calculate mean of variance and other values
                values_all_mean = numpy.mean(values_all,0)
                #calculate standard error from mean variance
                values_all_mean[:,std_isi_idx] = numpy.sqrt(values_all_mean[:,std_isi_idx]/self.repetitions)
                #write the averaged values in the initial_values dictionary
                mean_dict_all = []
                for vals in values_all_mean:
                    mean_dict = dict(zip(value_keys, vals))
                    mean_dict_all.append(mean_dict)
                self.initial_data.update(dict(zip(neuron_coords, mean_dict_all)))
                self.run_time += time.time() - t_start

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
