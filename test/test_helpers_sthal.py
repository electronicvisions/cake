#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the sthal helper.
"""

import unittest
import pickle
import os
import tempfile

from pycake.helpers.sthal import StHALContainer, UpdateAnalogOutputConfigurator
from pyhalbe import Coordinate as C


class TestSthalHelper(unittest.TestCase):
    def test_init(self):
        coord_wafer = C.Wafer(4)
        coord_hicann = C.HICANNOnWafer(C.Enum(365))
        sthal = StHALContainer(coord_wafer, coord_hicann)
        p = pickle.dumps(sthal)
        s2 = pickle.loads(p)
        self.assertIsInstance(s2, StHALContainer)

    def test_dump(self):
        coord_wafer = C.Wafer(4)
        coord_hicann = C.HICANNOnWafer(C.Enum(365))
        filehandle, filename = tempfile.mkstemp(suffix=".xml.gz")
        self.addCleanup(os.remove, filename)
        sthal = StHALContainer(coord_wafer, coord_hicann, dump_file=filename)
        sthal.connect()
        sthal.write_config()
        sthal.disconnect()

    def test_configurator(self):
        configurator = UpdateAnalogOutputConfigurator()
        del configurator


if __name__ == "__main__":
    unittest.main()
