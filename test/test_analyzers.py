#!/usr/bin/env python

import unittest
import numpy as np

import pycake.analyzer


class TestAnalyzers(unittest.TestCase):

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
        a = pycake.analyzer.PeakAnalyzer()
        # TODO

    def test_I_gl_Analyzer(self):
        #a = pycake.analyzer.I_gl_Analyzer()
        # TODO
        pass

    def test_I_pl_Analyzer(self):
        a = pycake.analyzer.I_pl_Analyzer()
        # TODO


if __name__ == '__main__':
    unittest.main()
