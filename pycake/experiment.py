import pylogging
import time
import numpy

class BaseExperiment(object):
    """ Takes a list of measurements and analyzers.
        Then, it runs the measurements and analyzes them.
        Traces can be saved to hard drive or discarded after analysis.

        Experiments can be continued after a crash by just loading and starting them again.

        Args:
            measurements: list of Measurement objects
            analyzer: list of analyzer objects (or just one)
            save_traces: (True|False) should the traces be saved to HDD
    """

    logger = pylogging.get("pycake.experiment")

    #TODO remove save_traces
    #TODO hold neurons
    def __init__(self, measurements, analyzer, save_traces):
        self.measurements = measurements
        self.analyzer = analyzer
        self.results = []
        self.save_traces = save_traces

    def run(self):
        """Run the experiment and process results."""
        return list(self.iter_measurements())

    def iter_measurements(self):
        i_max = len(self.measurements)
        for i, measurement in enumerate(self.measurements):
            if not measurement.done:
                self.logger.INFO("Running measurement {}/{}".format(i+1, i_max))
                results = measurement.run_measurement(self.analyzer)
                self.results.append(results)
                yield True # Used to save state of runner 
            else:
                self.logger.INFO("Measurement {}/{} already done. Going on with next one.".format(i+1, i_max))
                yield False
        return

    def get_measurement(self, i):
        try:
            return self.measurements[i]
        except IndexError:
            return None

    def get_parameters_and_results(self, neuron, parameters, result_keys):
        """ Read out parameters and result for the given neuron.
        Shared parameters are read from the corresponding FGBlock.

            Args:
                neuron: parameters matching this neuron
                parameter: [list] parameters to read (neuron_parameter or
                           shared_parameter)
                result_keys: [list] result keys to append
            Return:
                2D numpy array: each row contains the containing parameters
                    in the requested order, followed by the requested keys
        """
        values =[]
        for measurement, result in zip(self.measurements, self.results):
            value = measurement.get_parameters(neuron, parameters)
            for key in result_keys:
                value.append(result[neuron][key])
            values.append(value)
        return numpy.array(values)
