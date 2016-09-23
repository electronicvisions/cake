#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the calibtic helper.
"""

import unittest
import pickle
import os
import shutil
import tempfile

import pylogging
import pycalibtic
from pycake.helpers.calibtic import Calibtic, create_pycalibtic_transformation
from pyhalbe import Coordinate as C
from pyhalbe.HICANN import neuron_parameter, shared_parameter


class TestCalibticHelper(unittest.TestCase):
    def setUp(self):
        basedir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, basedir)
        self.basedir = basedir

        wafer = C.Wafer(0)
        hicann = C.HICANNOnWafer(C.Enum(94))
        self.calibtic = Calibtic(self.basedir, wafer, hicann)

    def test_polynomial(self):
        # 3 + 2*x + 1*x^2
        poly = create_pycalibtic_transformation([3, 2, 1])
        self.assertEqual(poly.apply(1), 6)
        self.assertEqual(poly.apply(3), 18)

    def test_get_backend(self):
        """an invalid backend type has to raise a ValueError"""
        c = self.calibtic
        # initializing magic type should raise
        self.assertRaises(ValueError, c.get_backend, "magic")

    def test_pickle(self):
        c = self.calibtic
        # try pickling and unpickling
        p = pickle.dumps(c)
        c2 = pickle.loads(p)
        self.assertIsInstance(c2, Calibtic)

    def test_readout_shift(self):
        c = self.calibtic
        neuron = C.NeuronOnHICANN(C.Enum(482))

        # read without data
        c.clear_calibration()
        c._loaded = False  # reload
        self.assertEqual(c.get_readout_shift(neuron), 0.0)

        # read with data, but no readout shift
        trafo = create_pycalibtic_transformation([4,2,3])
        data = {c: trafo for c in C.iter_all(C.NeuronOnHICANN)}
        c.write_calibration(neuron_parameter.V_t, data, False, "normal", "normal", "normal")
        self.assertEqual(c.get_readout_shift(neuron), 0.0)

        # write readout shift, read
        trafo = create_pycalibtic_transformation([4])
        data = {c: trafo for c in C.iter_all(C.NeuronOnHICANN)}
        c.write_calibration("readout_shift", data, False, "normal", "normal", "normal")
        self.assertEqual(c.get_readout_shift(neuron), 4.0)

    def test_clear(self):
        c = self.calibtic
        c.clear_calibration()

        # NeuronCalibration
        # same data for all neurons
        trafo = create_pycalibtic_transformation([4,2,3])
        data = {c: trafo for c in C.iter_all(C.NeuronOnHICANN)}
        c.write_calibration(neuron_parameter.V_t, data, False, "normal", "normal", "normal")
        c._loaded = False  # reload
        c.clear_one_calibration(neuron_parameter.V_t)

        # FGBlock
        trafo = create_pycalibtic_transformation([4, 2, 5])
        data = {c: trafo for c in C.iter_all(C.FGBlockOnHICANN)}
        c._loaded = False  # reload
        c.write_calibration(shared_parameter.V_reset, data, False, "normal", "normal", "normal")
        c.clear_one_calibration(shared_parameter.V_reset)

    def test_overwrite(self):
        c = self.calibtic
        trafo = create_pycalibtic_transformation([4,2,3])
        data = {c: trafo for c in C.iter_all(C.NeuronOnHICANN)}
        c.write_calibration(neuron_parameter.V_t, data, False, "normal", "normal", "normal")
        c.write_calibration(neuron_parameter.V_t, data, False, "normal", "normal", "normal")
        c.clear_calibration()

    def test_apply(self):
        c = self.calibtic
        c.clear_calibration()
        c._loaded = False  # reload

        neuron = C.NeuronOnHICANN(C.Enum(27))
        parameter = neuron_parameter.V_t
        calib = pycalibtic.NeuronCalibrationParameters.Calibrations.V_t
        calib_missing = pycalibtic.NeuronCalibrationParameters.Calibrations.E_l

        # write data
        coeffs = [4, 2, 3]
        trafos = create_pycalibtic_transformation(coeffs)
        data = {c: trafos for c in C.iter_all(C.NeuronOnHICANN)}
        c.write_calibration(parameter, data, False, "normal", "normal", "normal")

        # test apply with data
        c._loaded = False  # reload
        ncal = c.get_calibration(neuron)
        poly = ncal.at(calib)
        self.assertEqual(coeffs, list(poly.getData()))
        # 4 + 2*x + 3*x^2
        self.assertEqual(c.apply_calibration(1, parameter, neuron, False, "normal", "normal", "normal"), 9)
        self.assertEqual(c.apply_calibration(2, parameter, neuron, False, "normal", "normal", "normal"), 20)
        self.assertEqual(c.apply_calibration(5, parameter, neuron, False, "normal", "normal", "normal"), 89)
        # over 1023
        self.assertEqual(c.apply_calibration(2000, parameter, neuron, False, "normal", "normal", "normal"), 1023)

        # same for FGBlock
        coeffs = [-6, 2, 3]
        trafos = create_pycalibtic_transformation(coeffs)
        data = {c: trafos for c in C.iter_all(C.FGBlockOnHICANN)}
        parameter = shared_parameter.V_reset
        c.write_calibration(parameter, data, False, "normal", "normal", "normal")
        block = C.FGBlockOnHICANN(C.Enum(2))
        c._loaded = False  # reload
        bcal = c.get_calibration(block)
        poly = bcal.at(parameter)
        self.assertEqual(coeffs, list(poly.getData()))
        # -6 + 2*x + 3*x^2
        self.assertEqual(c.apply_calibration(0, parameter, block, False, "normal", "normal", "normal"), 0)
        self.assertEqual(c.apply_calibration(1, parameter, block, False, "normal", "normal", "normal"), 0)
        self.assertEqual(c.apply_calibration(2, parameter, block, False, "normal", "normal", "normal"), 10)
        self.assertEqual(c.apply_calibration(5, parameter, block, False, "normal", "normal", "normal"), 79)

        c.clear_calibration()

    def test_init_nodir(self):
        """init backend in non-existing directory"""
        wafer = C.Wafer(0)
        hicann = C.HICANNOnWafer(C.Enum(315))
        nonexistdir = os.path.join(self.basedir, "newdir")
        self.assertFalse(os.path.exists(nonexistdir))
        c = Calibtic(nonexistdir, wafer, hicann)
        self.assertIsInstance(c, Calibtic)
        self.assertTrue(os.path.isdir(nonexistdir))


if __name__ == "__main__":
    from pysthal.command_line_util import init_logger
    init_logger(pylogging.LogLevel.WARN, [
        ("Default", pylogging.LogLevel.INFO),
    ])

    unittest.main()
