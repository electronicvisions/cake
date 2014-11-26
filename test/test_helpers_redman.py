#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import unittest
import pickle
from pycake.helpers.redman import Redman
from pyhalbe import Coordinate as C


class TestRedmanHelper(unittest.TestCase):
    def test_init(self):
        hicann = C.HICANNGlobal(C.Enum(280))
        r = Redman("/tmp", hicann)

        # initializing magic type should raise
        self.assertRaises(ValueError, r.init_backend, "magic")

        # try pickling and unpickling
        p = pickle.dumps(r)
        r2 = pickle.loads(p)
        del r2


if __name__ == "__main__":
    unittest.main()
