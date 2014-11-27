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


class TestCalibticHelper(unittest.TestCase):
    def setUp(self):
        self.basedir = tempfile.mkdtemp()
        #self.add_cleanup(shutil.rmtree, self.basedir)

    def test_polynomial(self):
        # 3 + 2*x + 1*x^2
        poly = create_pycalibtic_polynomial([3, 2, 1])
        self.assertEqual(poly.apply(1), 6)
        self.assertEqual(poly.apply(3), 18)

    def test_init(self):
        """init backend"""
        wafer = C.Wafer(0)
        hicann = C.HICANNOnWafer(C.Enum(94))
        c = Calibtic(self.basedir, wafer, hicann)

        # initializing magic type should raise
        self.assertRaises(ValueError, c.init_backend, "magic")

        # try pickling and unpickling
        p = pickle.dumps(c)
        c2 = pickle.loads(p)
        self.assertIsInstance(c2, Calibtic)

    def test_init_nodir(self):
        """init backend in non-existing directory"""
        wafer = C.Wafer(0)
        hicann = C.HICANNOnWafer(C.Enum(315))
        nonexistdir = os.path.join(self.basedir, "newdir")
        self.assertFalse(os.path.exists(nonexistdir))
        c = Calibtic(nonexistdir, wafer, hicann)
        self.assertIsInstance(c, Calibtic)
        self.assertTrue(os.path.isdir(nonexistdir))

    def tearDown(self):
        """cleanup"""
        shutil.rmtree(self.basedir)

if __name__ == "__main__":
    unittest.main()
