#!/usr/bin/env python
"""Test pycake.calibrationrunner module functionality."""

import unittest
import os
from pycake.calibrationrunner import CalibrationRunner
from pycake.config import Config
from pycake.helpers.calibtic import Calibtic
from pyhalbe.HICANN import shared_parameter
from pyhalbe.HICANN import neuron_parameter

class TestCalibrationRunner(unittest.TestCase):

    def test_finalize(self):

        # call finalize
        config_dir = os.path.dirname(os.path.realpath(__file__))
        config = Config(None, os.path.join(config_dir, "example_config.py"))
        runner = CalibrationRunner(config)
        runner.finalize()

        # load written calibration
        calibtic = Calibtic(runner.calibtic.path, *config.get_coordinates())

        # check if technical parameters are correctly stored
        for parameter in config.parameters['technical_parameters']:

            if isinstance(parameter, shared_parameter):
                coords = config.get_blocks()
            elif isinstance(parameter, neuron_parameter):
                coords = config.get_neurons()
            else:
                raise RuntimeError("parameter {} neither shared nor of type neuron".format(parameter))

            for coord in coords:

                cal = calibtic.get_calibration(coord)

                requested_val = config.parameters['base_parameters'][parameter].toDAC().value
                stored_val = cal.at(parameter).apply(-1)

                self.assertEqual(requested_val, stored_val)

if __name__ == '__main__':
    unittest.main()
