#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This test makes sure that currents and voltages are correctly converted to DAC
and checks whether DAC, Voltage and Current have a toDAC() function at the
same time.
"""

import unittest
from pycake.helpers.units import Current, Voltage, DAC


class TestUnitsHelper(unittest.TestCase):
    def test_Current(self):
        c = Current(100)
        self.assertEqual(c.toDAC().value, 41)

    def test_Voltage(self):
        v = Voltage(360)
        self.assertEqual(v.toDAC().value, 205)

    def test_DAC(self):
        d = DAC(3.14)
        self.assertEqual(d.toDAC().value, 3)


if __name__ == "__main__":
    unittest.main()
