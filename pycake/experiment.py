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

        # remove non-pickleable keys
        for key in ['logger']:
            if key in odict:
                del odict[key]
        return odict

    def __setstate__(self, dic):
        dic['logger'] = pylogging.get("pycake.experiment")
        self.__dict__.update(dic)

    def __init__(self, measurements, analyzer, save_traces):
        self.measurements = measurements
        self.analyzer = analyzer
        self.results = []
        self.save_traces = save_traces

        self.logger = pylogging.get("pycake.experiment")

    def create_analyzer(self, analyzer):
        return analyzer()

    def run_experiment(self):
        """Run the experiment and process results."""
        i_max = len(self.measurements)
        i=1
        for measurement in self.measurements:
            if not measurement.done:
                self.logger.INFO("{} - Running measurement {}/{}".format(time.asctime(), i, i_max))
                measurement.run_measurement()
                # TODO CK: parallel processing done here
                self.logger.INFO("{} - Analyzing measurement {}/{}".format(time.asctime(), i, i_max))
                results = {}
                for neuron in measurement.neurons:
                    t, v = measurement.get_trace(neuron)
                    results[neuron] = self.analyzer(t,v, neuron)
                self.results.append(results)

                if not self.save_traces:
                    measurement.clear_traces()
            else:
                self.logger.INFO("{} - Measurement {}/{} already done. Going on with next one.".format(time.asctime(), i, i_max))
            i+=1

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
        
