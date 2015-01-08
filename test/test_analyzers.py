#!/usr/bin/env python

import unittest
import numpy as np

import pycake.analyzer
from pycake.helpers.TraceAverager import TraceAverager


class TestAnalyzers(unittest.TestCase):
    def assertEqualNumpyArrays(self, A, B, message=None):
        """compares two numpy arrays for equality"""
        if message is None:
            message = "{} is not equal to {}".format(str(A), str(B))
        self.assertTrue(np.array_equal(A, B), message)

    def assertArrayElementsAlmostEqual(self, A, value, message=None):
        """compares every array element to value"""
        res = np.allclose(A, value)
        if message is None:
            message = "{} is not element-wise almost equal to {}".format(str(A), str(value))
        self.assertTrue(res, message)

    def test_mean_analyzer(self):
        """
        Test MeanOfTraceAnalyzer using a fixed voltage dataset.
        """

        # time should not matter, thus random
        time = np.random.random(10)

        voltage = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # neuron should not matter, thus false
        neuron = False

        a = pycake.analyzer.MeanOfTraceAnalyzer()
        res = a(time, voltage, neuron)

        # make sure analyzer returns all values
        for key in ["std", "max", "min", "mean"]:
            self.assertTrue(key in res)

        # check that calculated values are correct
        self.assertEqual(res["min"], 1)
        self.assertEqual(res["max"], 10)
        self.assertEqual(res["mean"], 5.5)
        self.assertAlmostEqual(res["std"], 2.872281323269)

    def test_peak_analyzer(self):
        """Test PeakAnalyzer
            voltage data is sawtooth
        """
        a = pycake.analyzer.PeakAnalyzer()
        time = np.linspace(0, 10, 50, False)
        # time = 0..9.8, 5 spikes:
        dt = 10./5
        frequency = 1./dt
        voltage = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]*5)
        slope_falling = -(voltage.max()-voltage.min())/time[1]
        slope_rising = (voltage.max()-voltage.min())/(dt-time[1])  # linear

        # array index of minimum/maximum
        minindex = np.array([0, 10, 20, 30, 40])
        maxindex = minindex + 9
        # drop first minimum and last maximum, these are not detected
        minindex = minindex[1:]
        maxindex = maxindex[:-1]

        # neuron should not matter, thus false
        neuron = False

        res = a(time, voltage, neuron)
        self.assertEqual(res["hard_min"], 1)
        self.assertEqual(res["hard_max"], 10)
        self.assertEqual(res["mean_min"], 1)
        self.assertEqual(res["mean_max"], 10)
        self.assertEqual(res["mean"], 5.5)
        self.assertEqualNumpyArrays(res["mintab"][:, 0], minindex)
        self.assertEqualNumpyArrays(res["maxtab"][:, 0], maxindex)
        self.assertAlmostEqual(res["std"], 2.87, places=2)
        self.assertAlmostEqual(res["frequency"], frequency)
        self.assertAlmostEqual(res["mean_dt"], dt)

        a = pycake.analyzer.PeakAnalyzer(True)
        res = a(time, voltage, neuron)
        # as above
        self.assertEqual(res["hard_min"], 1)
        self.assertEqual(res["hard_max"], 10)
        self.assertEqual(res["mean_min"], 1)
        self.assertEqual(res["mean_max"], 10)
        self.assertEqual(res["mean"], 5.5)
        self.assertEqualNumpyArrays(res["mintab"][:, 0], minindex)
        self.assertEqualNumpyArrays(res["maxtab"][:, 0], maxindex)
        self.assertAlmostEqual(res["std"], 2.87, places=2)
        self.assertAlmostEqual(res["frequency"], frequency)
        self.assertAlmostEqual(res["mean_dt"], dt)

        # additional assertions
        self.assertArrayElementsAlmostEqual(res["slopes_falling"], slope_falling)
        self.assertArrayElementsAlmostEqual(res["slopes_rising"], slope_rising)

    @unittest.skip("WIP")
    def test_I_gl_Analyzer(self):
        """analyze exponentially decaying trace"""

        time = np.linspace(0, 10, 150, False)
        single_voltage = np.exp(np.linspace(100, 0, 50))
        voltage = np.repeat(single_voltage, 3)

        # coordinates should not matter, thus false
        coord_neuron = False

        adc_freq = 1./time[1]
        trace_averager = TraceAverager(adc_freq)
        a = pycake.analyzer.I_gl_Analyzer(trace_averager)

        std = 0.1
        res = a(time, voltage, coord_neuron, std)
        print res
        # TODO continue test

    @unittest.skip("WIP")
    def test_I_pl_Analyzer(self):
        a = pycake.analyzer.I_pl_Analyzer()
        time = np.linspace(0, 10, 50, False)
        dt = time[1]
        voltage = np.array([1, 2, 4, 8, 16, 32, 0, 0, 0, 0]*5)
        tau_ref = 4*dt

        # neuron should not matter, thus false
        neuron = False

        res = a(time, voltage, neuron)
        self.assertAlmostEqual(res["tau_refrac"], tau_ref)


if __name__ == '__main__':
    unittest.main()
