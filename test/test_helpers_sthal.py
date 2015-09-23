#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the sthal helper.
"""

import unittest
import pickle
import os
import tempfile
import numpy

from pycake.helpers.sthal import StHALContainer
from pycake.config import Config
from pyhalbe import Coordinate as C
from pyhalbe.HICANN import L1Address
from PysthalTest import PysthalTest, hardware


class TestSthalHelper(unittest.TestCase):
    def test_init(self):
        """initialize StHAL container"""
        cfg = Config(None, {"coord_wafer" : C.Wafer(4),
                            "coord_hicann" : C.HICANNOnWafer(C.Enum(365))})
        sthal = StHALContainer(cfg)
        p = pickle.dumps(sthal)
        s2 = pickle.loads(p)
        self.assertIsInstance(s2, StHALContainer)

    def test_dump(self):
        """Use StHAL dump handle."""
        coord_wafer = C.Wafer(4)
        coord_hicann = C.HICANNOnWafer(C.Enum(365))
        coord_neuron = C.NeuronOnHICANN(C.Enum(65))
        coord_driver = C.SynapseDriverOnHICANN(C.Enum(2))
        l1address = L1Address(41)
        filehandle, filename = tempfile.mkstemp(suffix=".xml.gz")
        self.addCleanup(os.remove, filename)
        config_dir = os.path.dirname(os.path.realpath(__file__))
        cfg = Config("random-name", {"coord_wafer" : coord_wafer,
                                     "coord_hicann" : coord_hicann,
                                     "dump_file" : filename})
        sthal = StHALContainer(cfg)
        sthal.connect()
        sthal.disconnect()
        sthal.read_adc_status()
        sthal.disconnect()
        sthal.read_wafer_status()
        sthal.disconnect()
        sthal.write_config()
        sthal.switch_analog_output(coord_neuron)
        sthal.switch_analog_output(coord_neuron, l1address.value())
        rate = 1e4
        no_generators = 2
        sthal.stimulateNeurons(rate, no_generators, excitatory=True)
        sthal.enable_synapse_line(coord_driver, l1address, excitatory=True)
        sthal.disable_synapse_line(coord_driver)
        sthal.set_current_stimulus(9, 23, pulse_length=7)
        sthal.switch_current_stimulus_and_output(coord_neuron, l1address.value())
        #sthal.read_adc() # FIXME: FakeAnalogRecorder does not fully match real AnalogRecorder
        sthal.disconnect()


class TestSthalWithHardware(PysthalTest):
    """actually connect to hardware"""
    @hardware
    def test_connect(self):

        cfg = Config(None, {"coord_wafer": C.Wafer(self.WAFER),
                            "coord_hicann": C.HICANNOnWafer(self.HICANN)})

        sthal = StHALContainer(cfg)
        sthal.connect()
        #sthal.read_adc_status()
        #sthal.read_wafer_status()
        #sthal.write_config()
        sthal.disconnect()


class TestSimSthal(unittest.TestCase):
    def setUp(self):
        """initialize SimStHAL container"""
        # import here because dependencies are optional
        from pycake.helpers.sthal import SimStHALContainer
        coord_wafer = C.Wafer(4)
        self.coord_hicann = coord_hicann = C.HICANNOnWafer(C.Enum(365))
        cfg = Config(None, {"sim_denmem": ":0", "hicann_version": -1,
                            "coord_hicann":self.coord_hicann, "coord_wafer" : coord_wafer})
        self.sthal = SimStHALContainer(cfg)

    def test_pickle(self):
        """Try pickling and unpickling"""
        p = pickle.dumps(self.sthal)
        s2 = pickle.loads(p)
        self.assertIsInstance(s2, SimStHALContainer)

    def test_empty_functions(self):
        """These functions should do nothing."""
        sthal = self.sthal
        sthal.connect()
        sthal.disconnect()
        sthal.connect_adc()
        sthal.write_config()
        sthal.read_adc_status()

    def test_raising(self):
        """These functions should raise."""
        sthal = self.sthal
        self.assertRaises(NotImplementedError, sthal.dump_connect, "foo", "bar")

    def test_wafer_status(self):
        status = self.sthal.read_wafer_status()
        self.assertEqual(list(status.hicanns)[0], self.coord_hicann)

    def test_resample(self):
        """Test resampling of result data"""

        adc_interval = 1. / 96e6

        x = numpy.logspace(-2, 2, 100) * 1e-7
        y = numpy.linspace(0, 10, 100)

        result = dict(
            t=x,
            a=y,
            b=y,
            c=y)

        resampled = self.sthal.resample_simulation_result(result, adc_interval)

        for key in "a b c".split():
            self.assertEqual(
                len(resampled.index.values), len(resampled['a']),
                "re-sampled signal data has the same lenght as time")

        self.assertAlmostEqual(
            resampled.index.values[1] - resampled.index.values[0],
            adc_interval,
            "re-sampled interval is sufficiently close to adc sampling interval")

        intervals = numpy.diff(resampled.index.values)
        self.assertLess(
            (max(intervals) - min(intervals)) / numpy.mean(intervals),
            1e-10)


if __name__ == "__main__":
    PysthalTest.main()
