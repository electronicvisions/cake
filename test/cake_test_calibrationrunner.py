#!/usr/bin/env python
"""Test pycake.calibrationrunner module functionality."""

import os
import shutil
import tempfile
import unittest

from pycake.calibrationrunner import CalibrationRunner
from pycake.config import Config
from pycake.helpers.calibtic import Calibtic
from pyhalbe.HICANN import shared_parameter
from pyhalbe.HICANN import neuron_parameter

class TestCalibrationRunner(unittest.TestCase):

    def setUp(self):
        basedir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, basedir)
        self.basedir = basedir

    def load_config(self, filename):
        config_dir = os.path.dirname(os.path.realpath(__file__))
        config = Config(None, os.path.join(config_dir, filename))
        config.parameters['backend'] = self.basedir
        config.parameters['folder'] = self.basedir
        return config

    def test_finalize(self):
        # call finalize
        config = self.load_config("example_config_A.py")
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


    def test_save_experiment_results(self):
        """
        Check that save_experiment_results can be called with none unit step
        data
        """
        runner = CalibrationRunner(self.load_config("example_config_C.py"))
        idx = runner.to_run.index('I_gl_charging')
        unit = runner.create_or_load_unit(idx)
        unit.save_experiment_results()


if __name__ == '__main__':
    unittest.main()
