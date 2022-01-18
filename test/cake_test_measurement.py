#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Measurement
"""


import unittest
import shutil
import tempfile
import os
import numpy
import pandas

from pycake.config import Config
from pyhalco_common import iter_all, Enum
import pyhalco_hicann_v2 as C
try:
    from pycake.helpers.sim import SimStHALContainer
except ImportError:
    # if used without-sim use dummy sthal container
    def SimStHALContainer(*args, **kwargs):
        pass
import pycake.measure
import pycake.analyzer

class TestMeasurement(unittest.TestCase):

    def setUp(self):
        self.basedir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.basedir)

        rng_trace = numpy.random
        rng_trace.seed(3215)

        self.neurons = [neuron for neuron in iter_all(C.NeuronOnHICANN)]
        traces = [rng_trace.random(3000) for _ in self.neurons]
        times = [numpy.linspace(0., 1e-3, len(traces[0]))  for _ in self.neurons]
        self.testdata = {nn: pandas.DataFrame({'v': v}, index=t) for nn,t,v in zip(self.neurons, times, traces)}

        cfg = Config(None, {"sim_denmem": ":0", "hicann_version": -1,
                            "coord_hicann": C.HICANNOnWafer(Enum(365)), "coord_wafer" : C.Wafer(4)})
        self.sthal = SimStHALContainer(cfg)

    def test_save_traces(self):
        # test if traces can be saved in a new directory
        temp_trace_path = os.path.join(self.basedir, "newtrace.h5")
        self.assertFalse(os.path.exists(temp_trace_path))
        measurement = pycake.measure.ADCMeasurement(self.sthal, self.neurons)
        measurement.save_traces(temp_trace_path)
        # store some data, save neurons to check
        trace_keys = []
        for ii, (neuron, trace) in enumerate(self.testdata.items()):
            if ii < 10:
                measurement.traces[neuron] = trace
                trace_keys.append('/trace_{:0>3}'.format(neuron.toEnum().value()))
        with pandas.HDFStore(temp_trace_path) as store:
            self.assertListEqual(sorted(store.keys()), sorted(trace_keys))
        # traces could be saved correctly, now test what happens if the store is corrupt
        os.remove(temp_trace_path)
        # just write some string to the file that it can not read
        with open(temp_trace_path, 'wb') as handle:
            handle.write(b"\x0a\x1b\x2c")
        measurement2 = pycake.measure.ADCMeasurement(self.sthal, self.neurons)
        measurement2.save_traces(temp_trace_path)
        trace_keys = []
        for ii, (neuron, trace) in enumerate(self.testdata.items()):
            if 20 < ii and ii < 30:
                measurement2.traces[neuron] = trace
                trace_keys.append('/trace_{:0>3}'.format(neuron.toEnum().value()))
        self.assertTrue(os.path.exists(temp_trace_path))
        # try to open the store again and check for the keys
        with pandas.HDFStore(temp_trace_path) as store:
            self.assertListEqual(sorted(store.keys()), sorted(trace_keys))

if __name__ == "__main__":
    import pylogging
    from pysthal.command_line_util import init_logger
    init_logger(pylogging.LogLevel.WARN, [
        ("Default", pylogging.LogLevel.INFO),
    ])

    unittest.main()

