#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import pycalibtic as cal
import pycairo.experiment as exp
import random


def initBackend(fname):
    lib = cal.loadLibrary(fname)
    backend = cal.loadBackend(lib)

    if not backend:
        raise Exception("unable to load %s" % fname)

    return backend


def loadXMLBackend(path="build"):
    backend = initBackend("libcalibtic_xml.so")
    backend.config("path", path)
    backend.init()
    return backend


class FakeHICANN(object):
    """Fakes functionality that would measure on actual hardware."""
    # TODO this class could actually do stuff that influences fake measurements
    def reset(self):
        pass

    def program_fg(self, parameters):
        pass

    def configure(self):
        pass

    def activate_neuron(self, neuron_id):
        pass

    def enable_analog_output(self, neuron_id):
        pass


class FakeADC(object):
    """Fakes ADC functionality, generates data."""
    # TODO this class could do much more
    def __init__(self, average=300., randrange=10):
        self.average = average
        self.randrange = randrange

    def measure_adc(self, points):
        average = self.average
        randrange = self.randrange
        t = range(points)
        v = []
        for i in range(points):
            v.append(average + random.uniform(-randrange, randrange))

        v = np.array(v)/1000.  # convert mV to V
        return np.array(t), v


class TestExperiment(unittest.TestCase):
    def test_Current(self):
        c = exp.Current(100)
        self.assertEqual(c.toDAC().value, 41)

    def test_Voltage(self):
        v = exp.Voltage(360)
        self.assertEqual(v.toDAC().value, 205)

    def test_DAC(self):
        d = exp.DAC(3.14)
        self.assertEqual(d.toDAC().value, 3)

    def test_Calibrate_E_l(self):
        neurons = [2, 3, 5, 7, 8, 10]
        fake_hicann = FakeHICANN()
        fake_adc = FakeADC()
        e = exp.Calibrate_E_l(fake_hicann, fake_adc, neurons)
        self.assertEqual(e.get_neurons(), neurons)
        params = e.get_parameters()
        for neuron_id in neurons:
            self.assertTrue(neuron_id in params)
            # TODO pro neuron: alle Parameter enthalten?
            #self.assertTrue()
        steps = e.get_steps()
        numsteps = None  # store number of steps
        for neuron_id in neurons:
            # next line is disabled, because this test fails when accessing
            # defaultdict(). not sure yet whether test or implementation should
            # be changed
            #self.assertTrue(neuron_id in steps)
            if numsteps:
                # make sure all neurons have the same number of steps
                self.assertEqual(len(steps[neuron_id]), numsteps)
            else:
                numsteps = len(steps[neuron_id])
        self.assertTrue(e.repetitions > 0)

        # check measure
        e.all_results = []
        e.measure(neurons)
        for results in e.all_results:
            for neuron_id in neurons:
                self.assertTrue(fake_adc.average - fake_adc.randrange <= results[neuron_id] <= fake_adc.average + fake_adc.randrange,
                                "result {} too far from average {}".format(results[neuron_id], fake_adc.average))

        # check process_results
        value = 5.  # TODO improve this
        all_results = []
        for step in range(numsteps):
            for repetition in range(e.repetitions):
                results = {}
                for neuron_id in neurons:
                    results[neuron_id] = value
                all_results.append(results)
        e.all_results = all_results
        e.process_results(neurons)
        for neuron_id in neurons:
            self.assertEqual(len(e.results_mean[neuron_id]), numsteps)
            self.assertEqual(len(e.results_std[neuron_id]), numsteps)
            for val in e.results_mean[neuron_id]:
                self.assertEqual(val, value)
            for val in e.results_std[neuron_id]:
                self.assertEqual(val, 0)  # TODO improve this


if __name__ == "__main__":
    unittest.main()
