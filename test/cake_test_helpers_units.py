#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This test makes sure that currents and voltages are correctly converted to DAC
and checks whether DAC, Volt and Ampere have a toDAC() function at the
same time.
"""

import unittest
from pycake.helpers.units import (Ampere, Volt, DAC,
                                  linspace_voltage, linspace_current)


class TestUnitsHelper(unittest.TestCase):
    def test_Ampere(self):
        c = Ampere(100e-9)
        self.assertEqual(c.toDAC().value, 41)
        self.assertIsInstance(repr(c), str)

        # values outside range
        self.assertRaises(ValueError, Ampere, -1e-9)
        self.assertRaises(ValueError, Ampere, 2.501e-6)

    def test_Volt(self):
        v = Volt(0.36)
        self.assertEqual(v.toDAC().value, 205)
        self.assertIsInstance(repr(v), str)

        # values outside range
        self.assertRaises(ValueError, Volt, -1e-3)
        self.assertRaises(ValueError, Volt, 1.801)

    def test_DAC(self):
        d = DAC(3.14)
        self.assertEqual(d.toDAC().value, 3)
        self.assertIsInstance(repr(d), str)

        # values outside range
        self.assertRaises(ValueError, DAC, -1)
        self.assertRaises(ValueError, DAC, 1024)

        self.assertRaises(TypeError, d._check, 1.5)

    def test_DACtoOther(self):
        self.assertEqual(DAC(0).toAmpere().value, 0.)
        self.assertEqual(DAC(0).toVolt().value, 0.)
        self.assertEqual(DAC(1023).toAmpere().value, 2.5e-6)
        self.assertEqual(DAC(1023).toVolt().value, 1.8)

    def test_Linspace(self):
        val = [v.value for v in linspace_voltage(0.1, 0.3, 3)]
        self.assertEqual([0.1, 0.2, 0.3], val)

        val = [c.value for c in linspace_current(0.1e-6, 0.3e-6, 3)]
        self.assertEqual([0.1e-6, 0.2e-6, 0.3e-6], val)


if __name__ == "__main__":
    unittest.main()
