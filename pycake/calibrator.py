"""Manages experiments

"""
import numpy as np
import pylogging
from collections import defaultdict
import pyhalbe
import pycalibtic
from pyhalbe.Coordinate import NeuronOnHICANN, FGBlockOnHICANN
from pycake.helpers.calibtic import init_backend as init_calibtic
from pycake.helpers.redman import init_backend as init_redman
import pyredman as redman
from pycake.helpers.units import Current, Voltage, DAC
from pycake.helpers.trafos import HWtoDAC, DACtoHW, HCtoDAC, DACtoHC, HWtoHC, HCtoHW
from pycake.helpers.WorkerPool import WorkerPool
import pycake.helpers.misc as misc
import pycake.helpers.sthal as sthal
from pycake.measure import Measurement
import pycake.analyzer

# Import everything needed for saving:
import pickle
import time
import os
import bz2
import imp
import copy

# shorter names
Coordinate = pyhalbe.Coordinate
Enum = Coordinate.Enum
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter


class Calibrator(object):
    """ Creates experiments for one target parameter, merges them and does the fits.

    Args:
        target_parameter: neuron_parameter or shared_parameter
        feature: string containing the feature name (e.g. "E_l" or "g_l")
        experiments: list containing all experiments
    """
    def __init__(self, target_parameter, feature, experiments):
        self.target_parameter = target_parameter
        if not isinstance(experiments, list):
            experiments = [experiments]
        self.experiments = experiments
        self.feature = feature

        self.analyzer = getattr(pycake.analyzer, '{}_Analyzer'.format(feature))

        # check if all experiments fit to the given target parameter
        for ex in experiments:
            assert ex.target_parameter == self.target_parameter, 'Experiments do not fit together'

    def check_for_same_config(self, measurements):
        """ Checks if measurements have the same config.
            Only the target parameter is allowed to be different.
        """
        pass

    def merge_experiments(self):
        """ Merges all experiments into one result dictionary
        """
        # measured_values = {neuron1: {step1: [value1, value2], step2: [value1, value2]},
        #                    neuron2: {step1: [value1, .....}
        # {neuron: [(step1, value),(step2, value), (step1, value)]}
        measured_values = defaultdict(list)
        for ex in self.experiments:
            for m in ex.measurements:
                # Create analyzer
                analyzer = self.analyzer(m)

                # Get the step (x) value
                # To do this, we need to get the value for all neurons and check if all neurons were set to the same
                # When this is ensured, we can chose any value from the list.
                step_values = m.get_parameter(target_parameter, m.neurons).values()
                assert len(set(step_values)) == 1, "Neurons were not set to the same value."
                step_value = step_values.values()[0]

                # Get analyzed (y) values:
                analyzed_values = analyzer.analyze()

                # Now add x and y values to the right measured_values entry
                for neuron, analyzed_value in analyzed_values.iteritems():
                    x_value = prepare_x(step_value)
                    y_value = prepare_y(analyzed_value)
                    measured_values[neuron].append((x_value, y_value))

        return measured_values

    def calibrate(self):
        """ Takes averaged experiments and does the fits (if it is not a test measurement)
        """
        self.average_experiments()

    def prepare_x(self, x):
        """ Prepares x value for fit
        """
        return x

    def prepare_y(self, y):
        """ Prepares y value for fit
        """
        return y

    def do_fit(self, x, y):
        """ Fits a curve to results of one neuron
        """

    def store_calibration(self):
        """ Stores calibration into backend.
        """

