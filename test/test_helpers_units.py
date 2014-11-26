#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This test makes sure that currents and voltages are correctly converted to DAC
and checks whether DAC, Voltage and Current have a toDAC() function at the
same time.
"""

import unittest
from pycake.helpers.units import (Current, Voltage, DAC,
                                  linspace_voltage, linspace_current)


class TestUnitsHelper(unittest.TestCase):
    def test_Current(self):
        c = Current(100)
        self.assertEqual(c.toDAC().value, 41)
        self.assertIsInstance(repr(c), str)

        # values outside range
        self.assertRaises(ValueError, Current, -1)
        self.assertRaises(ValueError, Current, 2501)

    def test_Voltage(self):
        v = Voltage(360)
        self.assertEqual(v.toDAC().value, 205)
        self.assertIsInstance(repr(v), str)

        # values outside range
        self.assertRaises(ValueError, Voltage, -1)
        self.assertRaises(ValueError, Voltage, 1801)

    def test_DAC(self):
        d = DAC(3.14)
        self.assertEqual(d.toDAC().value, 3)
        self.assertIsInstance(repr(d), str)

        # values outside range
        self.assertRaises(ValueError, DAC, -1)
        self.assertRaises(ValueError, DAC, 1024)

        self.assertRaises(TypeError, d._check, 1.5)

    def test_DACtoOther(self):
        self.assertEqual(DAC(0).toCurrent().value, 0.)
        self.assertEqual(DAC(0).toVoltage().value, 0.)
        self.assertEqual(DAC(1023).toCurrent().value, 2500.)
        self.assertEqual(DAC(1023).toVoltage().value, 1800.)

    def test_Linspace(self):
        val = [v.value for v in linspace_voltage(100, 300, 3)]
        self.assertEqual([100., 200., 300.], val)

        val = [c.value for c in linspace_current(100, 300, 3)]
        self.assertEqual([100., 200., 300.], val)


if __name__ == "__main__":
    unittest.main()
