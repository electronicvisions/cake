import pylogging
import time

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
