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
        res = a(neuron=neuron, t=time, v=voltage)

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

        res = a(neuron=neuron, t=time, v=voltage)
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
        res = a(neuron=neuron, t=time, v=voltage)
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
        res = a(neuron=coord_neuron, t=time, v=voltage)
        print res
        # TODO continue test

    def test_ISI_Analyzer(self):
        a = pycake.analyzer.ISI_Analyzer()
        a2 = pycake.analyzer.ISI_Analyzer2()

        # neuron should not matter, thus false
        neuron = False

        time = np.linspace(0, 10, 60, False)
        dt = time[1]

        voltage = np.array([1, 2, 4, 8, 16, 32]*10)
        res = a(neuron=neuron, t=time, v=voltage)
        res2 = a2(neuron=neuron, t=time, v=voltage)
        isi0 = res["mean_isi"]
        isi2_0 = res2["mean_isi"]
        self.assertAlmostEqual(isi0, 6*dt)
        self.assertAlmostEqual(isi2_0, 6*dt)
        self.assertAlmostEqual(res2["std_isi"], 0)

        voltage = np.array([1, 2, 4, 8, 16, 32, 0, 0, 0, 0, 0, 0]*5)
        actual_tau_ref = 6*dt
        res = a(neuron=neuron, t=time, v=voltage)
        res2 = a2(neuron=neuron, t=time, v=voltage)
        isi_with_tau = res["mean_isi"]
        isi_with_tau2 = res2["mean_isi"]
        tau_ref = isi_with_tau - isi0
        tau_ref2 = isi_with_tau2 - isi2_0
        self.assertAlmostEqual(tau_ref, actual_tau_ref)
        self.assertAlmostEqual(tau_ref2, actual_tau_ref)
        self.assertAlmostEqual(res2["std_isi"], 0)


if __name__ == '__main__':
    unittest.main()
