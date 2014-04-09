import numpy as np
import pylogging
from collections import defaultdict
import pyhalbe
import pycalibtic
from pyhalbe.Coordinate import NeuronOnHICANN, FGBlockOnHICANN

# Import everything needed for saving:
import pickle
import time


class Analyzer(object):
    """ Takes a measurement and analyses it.
    """
    def __init__(self, measurement):
        self.measurement = measurement

    def analyze(self):
        """ Returns a dictionary of results:
            {neuron: value}
        """

class E_l_Analyzer(Analyzer):
    """ Analyzes traces for E_l measurement.
    """
    def analyze(self):
        m = self.measurement
        results = {}

        for neuron in m.neurons:
            t,v = m.trace[neuron]
            results[neuron] = np.mean(v)

        return results
