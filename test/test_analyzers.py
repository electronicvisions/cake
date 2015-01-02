#!/usr/bin/env python

import unittest
import numpy as np

import pycake.analyzer


class TestAnalyzers(unittest.TestCase):
    def assertEqualNumpyArrays(self, A, B, message=None):
        """compares two numpy arrays for equality"""
        if message is None:
            message = "{} is not equal to {}".format(str(A), str(B))
        self.assertTrue(np.array_equal(A, B), message)

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
        voltage = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]*5)

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
        # TODO test remaining functionality

        a = pycake.analyzer.PeakAnalyzer(True)
        res = a(time, voltage, neuron)

    def test_I_gl_Analyzer(self):
        #a = pycake.analyzer.I_gl_Analyzer()
        # TODO
        pass

    def test_I_pl_Analyzer(self):
        a = pycake.analyzer.I_pl_Analyzer()
        # TODO


if __name__ == '__main__':
    unittest.main()
