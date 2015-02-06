#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test the sthal helper.
"""

import unittest
import pickle
import os
import tempfile

from pycake.helpers.sthal import StHALContainer
from pycake.helpers.sthal import SimStHALContainer
from pycake.config import Config
from pyhalbe import Coordinate as C
from pyhalbe.HICANN import L1Address
from PysthalTest import PysthalTest, hardware


class TestSthalHelper(unittest.TestCase):
    def test_init(self):
        """initialize StHAL container"""
        coord_wafer = C.Wafer(4)
        coord_hicann = C.HICANNOnWafer(C.Enum(365))
        sthal = StHALContainer(coord_wafer, coord_hicann)
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
        sthal = StHALContainer(coord_wafer, coord_hicann, dump_file=filename)
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
        sthal.read_adc()
        sthal.disconnect()


class TestSthalWithHardware(PysthalTest):
    """actually connect to hardware"""
    @hardware
    def test_connect(self):
        coord_wafer = C.Wafer(self.WAFER)
        coord_hicann = C.HICANNOnWafer(self.HICANN)
        sthal = StHALContainer(coord_wafer, coord_hicann)
        sthal.connect()
        #sthal.read_adc_status()
        #sthal.read_wafer_status()
        #sthal.write_config()
        sthal.disconnect()


class TestSimSthal(unittest.TestCase):
    def setUp(self):
        """initialize SimStHAL container"""
        coord_wafer = C.Wafer(4)
        self.coord_hicann = coord_hicann = C.HICANNOnWafer(C.Enum(365))
        cfg = Config(None, {"sim_denmem": ":0", "hicann_version": -1})
        self.sthal = SimStHALContainer(coord_wafer, coord_hicann, config=cfg)

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


if __name__ == "__main__":
    PysthalTest.main()
