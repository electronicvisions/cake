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
    def __call__(self, t, v, neuron):
        """ Returns a dictionary of results:
            {neuron: value}
        """
        raise NotImplemented

class MeanOfTraceAnalyzer(Analyzer):
    """ Analyzes traces for E_l measurement.
    """
    def __call__(self, t, v, neuron):
        return { "mean" : np.mean(v),
                 "std"  : np.std(v)}

