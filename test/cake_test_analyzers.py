#!/usr/bin/env python

import unittest
import numpy as np

import pycake.analyzer
from pycake.helpers.TraceAverager import TraceAverager

import pyhalco_hicann_v2
import pyhalco_common
import pickle

import pandas

def convert_old_pickle(d):
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = convert_old_pickle(v)
        elif isinstance(v, bytes):
            v = v.decode()
        if isinstance(k, bytes):
            k = k.decode()
        res[k] = v
    return(res)

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

        trace = pandas.DataFrame({'v': voltage}, index=time)

        res = a(neuron=neuron, trace=trace)

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

        trace = pandas.DataFrame({'v': voltage}, index=time)

        res = a(neuron=neuron, trace=trace)
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
        res = a(neuron=neuron, trace=trace)
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

        # If there is only one peak in the trace, analyzer should return dt = inf
        t = np.arange(10)
        v = [1, 2, 3, 4, 5, 1, 2, 3, 3, 3]
        trace = pandas.DataFrame({'v': v}, index=t)
        result = a(neuron=neuron, trace=trace)
        self.assertTrue(np.isinf(result['mean_dt']))

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
        res = a(neuron=coord_neuron, trace=trace)
        print(res)
        # TODO continue test

    def test_V_reset_Analyzer(self):
        """ Test baseline detection in V_reset_Analyzer
        """
        a = pycake.analyzer.V_reset_Analyzer()

        # neuron should not matter, thus false
        neuron = False

        time = np.linspace(0, 10, 55, False)
        dt = time[1]

        voltage = np.array([1, 2, 3, 4, 5, 6, 1, 1, 1, 1, 1]*5)
        trace = pandas.DataFrame({'v': voltage}, index=time)
        res = a(neuron=neuron, trace=trace)
        self.assertAlmostEqual(res['baseline'], 1)
        self.assertAlmostEqual(res['mean_min'], 1)
        self.assertAlmostEqual(res['mean_max'], 6)

    def test_ISI_Analyzer(self):
        a = pycake.analyzer.ISI_Analyzer()

        # neuron should not matter, thus false
        neuron = False

        time = np.linspace(0, 10, 60, False)
        dt = time[1]

        voltage = np.array([1, 2, 4, 8, 16, 32]*10)
        trace = pandas.DataFrame({'v': voltage}, index=time)
        res = a(neuron=neuron, trace=trace)
        isi0 = res["mean_isi"]
        self.assertAlmostEqual(isi0, 6*dt)
        self.assertAlmostEqual(res["std_isi"], 0)

        voltage = np.array([1, 2, 4, 8, 16, 32, 0, 0, 0, 0, 0, 0]*5)
        trace = pandas.DataFrame({'v': voltage}, index=time)
        actual_tau_ref = 6*dt
        res = a(neuron=neuron, trace=trace)
        isi_with_tau = res["mean_isi"]
        tau_ref = isi_with_tau - isi0
        self.assertAlmostEqual(tau_ref, actual_tau_ref)
        self.assertAlmostEqual(res["std_isi"], 0)

    def test_MeanOfTraceAnalyzer_realtrace(self):
        """
        Test MeanOfTraceAnalyzer using a previously measured dataset.
        """
        neuron = False
        testdata = convert_old_pickle(pickle.load(open('/wang/data/calibration/testdata/testdata.p', 'rb'), encoding='bytes'))['E_l']

        # patch testdata
        testdata['result']['size'] = len(testdata["v"])

        analyzer = pycake.analyzer.MeanOfTraceAnalyzer()

        trace = pandas.DataFrame({'v': testdata["v"]}, index=testdata["t"])
        result = analyzer(neuron, trace=trace)

        for key in result:
            self.assertAlmostEqual(result[key], testdata['result'][key], places=2)

    def test_V_reset_realtrace(self):
        """
        Test V_reset_Analyzer using a previously measured dataset.
        """
        neuron = False
        testdata = convert_old_pickle(pickle.load(open('/wang/data/calibration/testdata/testdata.p', 'rb'), encoding='bytes'))['V_reset']
        analyzer = pycake.analyzer.V_reset_Analyzer()

        trace = pandas.DataFrame({'v': testdata["v"]}, index=testdata["t"])
        result = analyzer(neuron, trace=trace)

        for key in result:
            self.assertAlmostEqual(result[key], testdata['result'][key], places=2)

    def test_I_pl_realtrace(self):
        """
        Test I_pl_Analyzer using a previously measured dataset.
        """
        neuron = pyhalco_hicann_v2.NeuronOnHICANN(pyhalco_common.Enum(100))
        testdata = convert_old_pickle(pickle.load(open('/wang/data/calibration/testdata/testdata.p', 'rb'), encoding='bytes'))['I_pl']
        analyzer = pycake.analyzer.I_pl_Analyzer()
        data = testdata['initial_data'][neuron]
        data['t'] = testdata['t']
        data['v'] = testdata['v']
        trace = pandas.DataFrame({'v': data['v']}, index=data['t'])

        result = analyzer(neuron, trace=trace, additional_data=testdata['initial_data'])

        for key in result:
            self.assertAlmostEqual(result[key], testdata['result'][key], places=2)

    def test_I_gl_realtrace(self):
        """
        Test I_pl_Analyzer using a previously measured dataset.
        """
        neuron = pyhalco_hicann_v2.NeuronOnHICANN(pyhalco_common.Enum(100))
        testdata = convert_old_pickle(pickle.load(open('/wang/data/calibration/testdata/testdata.p', 'rb'), encoding='bytes'))['I_gl']
        analyzer = pycake.analyzer.I_gl_Analyzer()
        additional_data = {}
        additional_data['adc_freq'] = testdata['adc_freq']
        additional_data['current'] = 5
        additional_data['std'] = 0.004
        additional_data['save_mean'] = True

        trace = pandas.DataFrame({'v': testdata['v']}, index=testdata['t'])
        result = analyzer(neuron, trace=trace, additional_data=additional_data)

        for key in ['tau_m', 'V_rest', 'std_tau', 'reduced_chisquare', 'height']:
            self.assertAlmostEqual(result[key], testdata['result'][key], places=2)

    def test_adc_freq_analyzer(self):
        """ Test ADCFreq_Analyzer
        """
        v = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.9]*5)
        t = np.linspace(0, 1e-6, len(v))

        a = pycake.analyzer.ADCFreq_Analyzer()

        # Check if correct adc freq is detected and fail if unlikely freq is detected
        trace = pandas.DataFrame({'v': v}, index=t)
        res = a(trace=trace, bg_rate=13.7e6)
        self.assertAlmostEqual(res['adc_freq'], 13.7e6*7)
        self.assertRaises(RuntimeError, a, trace, 5e6)

    def test_simplepspanalyzer(self):
        """ Test SimplePSPAnalyzer

        with the following PSP:

                     +
                    + ++
                   +    ++
                  +       ++
                 +          ++
        --------+             +-----
        """

        # baseline of 1
        # peak at index 13 with value 7 (psp height=7)
        # rise to (psp height)/e at index 13
        # fall to (psp height)/e at index 20

        v = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1],  dtype='float')
        t = np.linspace(0., 1e-6, len(v)) # dT = 4e-8

        a = pycake.analyzer.SimplePSPAnalyzer()

        trace = pandas.DataFrame({'v': v}, index=t)

        # excitatory (1)

        res = a(pyhalco_hicann_v2.NeuronOnHICANN(), trace=trace)

        self.assertEqual(res['baseline'], 1)
        self.assertEqual(res['peakvalue'], 7)
        self.assertEqual(res['pspheight'], 6)
        self.assertEqual(res['risetime'], t[13]-t[10])
        self.assertEqual(res['falltime'], t[20]-t[13])
        self.assertEqual(res['psparea'], 51*4e-8)

        # excitatory (2)

        # use the same PSP as for excitatory (1)
        # make baseline negative
        v -= 3

        trace = pandas.DataFrame({'v': v}, index=t)

        res = a(pyhalco_hicann_v2.NeuronOnHICANN(), trace=trace)

        self.assertEqual(res['baseline'], -2)
        self.assertEqual(res['peakvalue'], 4)
        self.assertEqual(res['pspheight'], 6)
        self.assertEqual(res['risetime'], t[13]-t[10])
        self.assertEqual(res['falltime'], t[20]-t[13])
        self.assertEqual(res['psparea'], 51*4e-8)

        # inhibitory (1)

        # use the same PSP as for excitatory (2) but
        # mirror
        v *= -1

        trace = pandas.DataFrame({'v': v}, index=t)

        res = a(pyhalco_hicann_v2.NeuronOnHICANN(), trace=trace)

        self.assertEqual(res['baseline'], 2)
        self.assertEqual(res['peakvalue'], -4)
        self.assertEqual(res['pspheight'], -6)
        self.assertEqual(res['risetime'], t[13]-t[10])
        self.assertEqual(res['falltime'], t[20]-t[13])
        self.assertEqual(res['psparea'], 51*4e-8)

        # inhibitory (2)

        # use the same PSP as for inhibitory (1) but
        # make baseline negative
        v -= 4

        trace = pandas.DataFrame({'v': v}, index=t)

        res = a(pyhalco_hicann_v2.NeuronOnHICANN(), trace=trace)

        self.assertEqual(res['baseline'], -2)
        self.assertEqual(res['peakvalue'], -8)
        self.assertEqual(res['pspheight'], -6)
        self.assertEqual(res['risetime'], t[13]-t[10])
        self.assertEqual(res['falltime'], t[20]-t[13])
        self.assertEqual(res['psparea'], 51*4e-8)

if __name__ == '__main__':
    import pylogging
    for module in ["pycake.analyzer", "pycake.helper.TraceAverager"]:
        logger = pylogging.get(module)
        pylogging.set_loglevel(logger, pylogging.LogLevel.INFO)
        pylogging.append_to_cout(logger)

    unittest.main()
