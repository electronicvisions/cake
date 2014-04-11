"""Framework for experiments on the hardware.

Derive your custom experiment classes with functions for setup,
measurement and processing from BaseExperiment or child classes.
"""

import numpy as np
import pylogging
import pyhalbe
import pycalibtic
from pyhalbe.Coordinate import NeuronOnHICANN, FGBlockOnHICANN
from pycake.helpers.calibtic import init_backend as init_calibtic
from pycake.helpers.redman import init_backend as init_redman
import pyredman as redman
from pycake.helpers.units import Current, Voltage, DAC
import pycake.helpers.misc as misc
import pycake.helpers.sthal as sthal
from pycake.measure import Measurement

# Import everything needed for saving:
import time

# shorter names
Coordinate = pyhalbe.Coordinate
Enum = Coordinate.Enum
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter


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
    def __getstate__(self):
        """ Disable stuff from getting pickled that cannot be pickled.
        """
        odict = self.__dict__.copy()
        del odict['logger']
        del odict['progress_logger']
        return odict

    def __setstate__(self, dic):
        # TODO fix loading of pickled experiment. 
        # Right now, calibtic stuff is not loaded, but only empty variables are set
        self.__dict__.update(dic)

    def __init__(self, measurements, analyzer, save_traces):
        self.measurements = measurements
        self.analyzer = analyzer()
        self.results = []

        self.logger = pylogging.get("pycake.experiment")
        self.progress_logger = pylogging.get("pycake.experiment.progress")

    def run_experiment(self):
        """Run the experiment and process results."""
        for measurement in self.measurements:
            if not measurement.done():
                measurement.run_measurement()
                # TODO CK: parallel processing done here
                results = {}
                for neuron, (t,v) in measurement.traces.iteritems()
                    results[neuron] = self.analyzer(t,v, neuron)
                self.results.append(results)

                if not self.save_traces:
                    measurement.traces = [None] * len(measurement.neurons) # this ensures that measurement.done() is True

            self.save_state()
        
    def save_state(self):
        """ Pickles itself to a file.
        """
        # TODO implement this
        pass

    def measure(self):
        """ Perform measurements for a single step on one or multiple neurons.
            
            Appends a measurement to the experiment's list of measurements.
        """
        readout_shifts = self.get_readout_shifts(self.neurons)
        measurement = Measurement(self.sthal, self.neurons, readout_shifts)
        measurement.run_measurement()
        
