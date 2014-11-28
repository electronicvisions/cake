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

from pycake.helpers.calibtic import Calibtic, create_pycalibtic_polynomial
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
        poly = create_pycalibtic_polynomial([3, 2, 1])
        self.assertEqual(poly.apply(1), 6)
        self.assertEqual(poly.apply(3), 18)

    def test_init(self):
        """init backend"""
        c = self.calibtic
        # initializing magic type should raise
        self.assertRaises(ValueError, c.init_backend, "magic")

    def test_pickle(self):
        c = self.calibtic
        # try pickling and unpickling
        p = pickle.dumps(c)
        c2 = pickle.loads(p)
        self.assertIsInstance(c2, Calibtic)

    def test_readout_shift(self):
        c = self.calibtic
        neuron = C.NeuronOnHICANN(C.Enum(482))
        self.assertEqual(c.get_readout_shift(neuron), 0.0)

    def test_clear(self):
        c = self.calibtic
        c.clear_calibration()

        # NeuronCalibration
        # same data for all neurons
        data = {c: [4, 2, 3] for c in C.iter_all(C.NeuronOnHICANN)}
        c.write_calibration(neuron_parameter.V_t, data)
        c.clear_one_calibration(neuron_parameter.V_t)

        # FGBlock
        data = {c: [4, 2, 5] for c in C.iter_all(C.FGBlockOnHICANN)}
        c.write_calibration(shared_parameter.V_reset, data)
        c.clear_one_calibration(shared_parameter.V_reset)

    def test_overwrite(self):
        c = self.calibtic
        data = {c: [4, 2, 3] for c in C.iter_all(C.NeuronOnHICANN)}
        c.write_calibration(neuron_parameter.V_t, data)
        c.write_calibration(neuron_parameter.V_t, data)
        c.clear_calibration()

    def test_apply(self):
        c = self.calibtic
        data = {c: [4, 2, 3] for c in C.iter_all(C.NeuronOnHICANN)}
        c.write_calibration(neuron_parameter.V_t, data)
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
    unittest.main()
