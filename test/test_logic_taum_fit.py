#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test functionality of pycake/logic/taum_fit.py
"""

import unittest
import pycake.logic.taum_fit as tf
import numpy as np


class TestFit(unittest.TestCase):
    def test_instantiate(self):
        """Create class instances."""

        # arbitrary numbers
        voltage = 100e-6
        time = 100
        pulselength = 1e-9

        tf.TaumExperiment(time, voltage, pulselength)
# TODO test everything!


if __name__ == "__main__":
    unittest.main()
