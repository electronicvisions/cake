#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the SimStHALContainer helper.

This needs to be separate from the StHAL helper to allow build
configurations without simulation dependency.
"""

import unittest
import pickle
import numpy

from pycake.config import Config
from pyhalbe import Coordinate as C
from PysthalTest import PysthalTest


class TestSimSthal(unittest.TestCase):
    def setUp(self):
        """initialize SimStHALContainer"""
        # import here because dependencies are optional
        from pycake.helpers.sim import SimStHALContainer
        coord_wafer = C.Wafer(4)
        self.coord_hicann = C.HICANNOnWafer(C.Enum(365))
        cfg = Config(None, {"sim_denmem": ":0", "hicann_version": -1,
                            "coord_hicann": self.coord_hicann, "coord_wafer" : coord_wafer})
        self.sthal = SimStHALContainer(cfg)

    def test_pickle(self):
        """Try pickling and unpickling"""
        # import here because dependencies are optional
        from pycake.helpers.sim import SimStHALContainer
        p = pickle.dumps(self.sthal)
        s2 = pickle.loads(p)
        self.assertIsInstance(s2, SimStHALContainer)

    def test_empty_functions(self):
        """These functions should do nothing."""
        sthal = self.sthal
        sthal.connect()
        sthal.disconnect()
        sthal.connect_adc()
        sthal.write_config()
        sthal.read_adc_status()

    def test_raising(self):
        """These functions should raise."""
        sthal = self.sthal
        self.assertRaises(NotImplementedError, sthal.dump_connect, "foo", "bar")

    def test_wafer_status(self):
        status = self.sthal.read_wafer_status()
        self.assertEqual(list(status.hicanns)[0], self.coord_hicann)

    def test_resample(self):
        """Test resampling of result data"""

        adc_interval = 1. / 96e6

        x = numpy.logspace(-2, 2, 100) * 1e-7
        y = numpy.linspace(0, 10, 100)

        result = dict(
            t=x,
            a=y,
            b=y,
            c=y)

        resampled = self.sthal.resample_simulation_result(result, adc_interval)

        for key in "a b c".split():
            self.assertEqual(
                len(resampled.index.values), len(resampled['a']),
                "re-sampled signal data has the same lenght as time")

        self.assertAlmostEqual(
            resampled.index.values[1] - resampled.index.values[0],
            adc_interval,
            "re-sampled interval is sufficiently close to adc sampling interval")

        intervals = numpy.diff(resampled.index.values)
        self.assertLess(
            (max(intervals) - min(intervals)) / numpy.mean(intervals),
            1e-10)


if __name__ == "__main__":
    PysthalTest.main()
