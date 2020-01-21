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
import pyhalco_hicann_v2 as C
from pyhalbe.HICANN import L1Address
from PysthalTest import PysthalTest, hardware

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


if __name__ == "__main__":
    PysthalTest.main()
