#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import numpy
import unittest
import tempfile
import os
from numpy.testing import assert_array_equal
from pyhalco_common import iter_all
from pyhalco_hicann_v2 import NeuronOnHICANN

from pycake.helpers.TracesOnDiskDict import RecordsOnDiskDict

class TestRecordsOnDiskDictHelper(unittest.TestCase):

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_file.close()
        print("create temp file: ", self.temp_file.name)
        self.addCleanup(os.remove, self.temp_file.name)

    def test_call(self):
        neurons = [n for n in iter_all(NeuronOnHICANN)][::30]

        data = dict((n, self.fake_record()) for n in neurons[::30])
        data2 = dict((n, self.fake_record()) for n in neurons[:10])

        records = RecordsOnDiskDict(*os.path.split(self.temp_file.name))
        self.assertEqual(records.fullpath, self.temp_file.name)

        for n, d in data.items():
            records[n] = d
        records.close()

        records = RecordsOnDiskDict(*os.path.split(self.temp_file.name))
        for n, d in data.items():
            for k, v in records[n].items():
                assert_array_equal(d[k], v)

        for n, d in data2.items():
            records[n] = d
        records.close()

        records = RecordsOnDiskDict(*os.path.split(self.temp_file.name))
        for n, d in data2.items():
            for k, v in records[n].items():
                assert_array_equal(d[k], v)
        records.close()


    @staticmethod
    def fake_record():
        size = 3000
        return {
            't': numpy.arange(size) + numpy.random.random(),
            'v': numpy.random.uniform(0.0, 1.8, size),
            'spikes': numpy.array([
                numpy.ones(size) * 4.0,
                numpy.cumsum(numpy.random.exponential(0.3, size))])
        }

if __name__ == "__main__":
    unittest.main()
