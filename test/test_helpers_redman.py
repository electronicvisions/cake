#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the redman helper.
"""

import unittest
import pickle
import os
import shutil
import tempfile

from pycake.helpers.redman import Redman
from pyhalbe import Coordinate as C


class TestRedmanHelper(unittest.TestCase):
    def setUp(self):
        self.basedir = tempfile.mkdtemp()

    def test_init(self):
        """init backend"""
        hicann = C.HICANNGlobal(C.Enum(94))
        r = Redman(self.basedir, hicann)

        # initializing magic type should raise
        self.assertRaises(ValueError, r.init_backend, "magic")

        # try pickling and unpickling
        p = pickle.dumps(r)
        r2 = pickle.loads(p)
        self.assertIsInstance(r2, Redman)

    def test_init_nodir(self):
        """init backend in non-existing directory"""
        hicann = C.HICANNGlobal(C.Enum(315))
        nonexistdir = os.path.join(self.basedir, "newdir")
        self.assertFalse(os.path.exists(nonexistdir))
        r = Redman(nonexistdir, hicann)
        self.assertIsInstance(r, Redman)
        self.assertTrue(os.path.isdir(nonexistdir))

    def test_defects(self):
        hicann = C.HICANNGlobal(C.Enum(11))
        n99 = C.NeuronOnHICANN(C.Enum(99))
        r = Redman(self.basedir, hicann)
        r.clear_defects()
        self.assertTrue(r.hicann_with_backend.neurons().has(n99))
        r.write_defects([n99])
        self.assertFalse(r.hicann_with_backend.neurons().has(n99))

        # force reloading
        r._loaded = False
        r.write_defects([n99])
        self.assertFalse(r.hicann_with_backend.neurons().has(n99))

    def tearDown(self):
        """cleanup"""
        shutil.rmtree(self.basedir)

if __name__ == "__main__":
    unittest.main()
