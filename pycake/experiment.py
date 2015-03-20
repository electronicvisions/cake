import pylogging
import time
import numpy
import copy
import os

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

    def get_measurement(self, i):
        try:
            return self.measurements[i]
        except IndexError:
            return None

    def append_measurement_and_result(self, measurement, result):
        self.measurements.append(measurement)
        self.results.append(result)

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
        for measurement, result in zip(self.measurements, self.results):
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
        import pandas
        names = [p.name for p in parameters]
        names.extend(result_keys)
        values = []
        for msr, result in zip(self.measurements, self.results):
            value = msr.get_parameters(neuron, parameters)
            nrn_results = result[neuron]
            value.extend(nrn_results.get(key, numpy.nan) for key in result_keys)
            values.append(numpy.array(value))
        return pandas.DataFrame(values, columns=names)

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
        """ Dummy f/or preparation measurements.
        """
        if len(self.initial_measurements) > 0:
            self.logger.INFO("Running initial measurements.")
            for measurement, analyzer in self.initial_measurements:
                self.initial_data.update(measurement.run_measurement(analyzer, None))

    def save_traces(self, path):
        for mid, measurement in enumerate(self.measurements_to_run):
            filename = os.path.join(path, "{}.hdf5".format(mid))
            measurement.save_traces(filename)

    def iter_measurements(self):
        self.run_initial_measurements()
        i_max = len(self.measurements_to_run)
        for i, measurement in enumerate(self.measurements_to_run):
            if not measurement.done:
                self.logger.INFO("Running measurement {}/{}".format(i+1, i_max))
                result = measurement.run_measurement(self.analyzer, self.initial_data)
                self.append_measurement_and_result(measurement, result)
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


# Compatibility for old pickels
BaseExperiment = SequentialExperiment


class IncrementalExperiment(Experiment):

    def __init__(self, initial_configuration, generator, analyzer):
        Experiment.__init__(self, analyzer)
        self.initial_configuration = initial_configuration
        self.generator = generator
        self.traces_folder = None

    def finished(self):
        # TODO@CK fix this
        return False

    def save_traces(self, path):
        self.traces_folder = path

    def iter_measurements(self):
        self.logger.INFO("Connecting to hardware and configuring.")
        sthal = self.initial_configuration
        sthal.write_config()
        i_max = len(self.generator)
        for i, (configurator, measurement) in enumerate(self.generator(sthal)):
            if self.traces_folder is not None:
                filename = os.path.join(self.traces_folder, "{}.hdf5".format(i))
                measurement.save_traces(filename)
            self.logger.INFO("Running measurement {}/{}".format(i+1, i_max))
            result = measurement.run_measurement(self.analyzer, configurator)
            measurement.sthal = copy.deepcopy(measurement.sthal)
            self.append_measurement_and_result(measurement, result)
            yield True # Used to save state of runner
