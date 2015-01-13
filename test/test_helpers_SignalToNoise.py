#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import unittest
import numpy as np
from pycake.helpers.SignalToNoise import SignalToNoise


class TestNoiseHelper(unittest.TestCase):
    def test_call(self):
        control = np.array([1, ]*100)
        adc_freq = 96e6
        bg_freq = 87
        stn = SignalToNoise(control, adc_freq, bg_freq)
        res = stn(np.random.randn(100))
        self.assertIsInstance(res, float)
        self.assertNotEqual(res, 0)
        # TODO check some actual value with actual data


if __name__ == "__main__":
    unittest.main()
