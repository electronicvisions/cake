#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import unittest
import pycake.logic.spikes as sp
import numpy as np


class TestSpikes(unittest.TestCase):
    def test_detect(self):
        """Test spike detection on "nice" trace (without noise)."""
        time = np.linspace(0, 1000, 1001)
        voltage = time % 100
        spike_times = sp.detect_spikes(time, voltage)
        for t in spike_times:
            # soft assertion
            self.assertTrue(voltage[t] > 96)
            ## hard assertion
            #self.assertEqual(voltage[t], 99.)

    def test_frequency(self):
        spikes = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(sp.spikes_to_frequency(spikes), 1)
        self.assertEqual(sp.spikes_to_frequency(2*spikes), 0.5)
        spikes = np.array([3, 3, 3, 3, 3, 3])
        self.assertEqual(sp.spikes_to_frequency(spikes), 0)


if __name__ == "__main__":
    unittest.main()
