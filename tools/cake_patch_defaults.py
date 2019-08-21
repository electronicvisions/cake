#!/usr/bin/env python
"""
Adds/resets default values to new or existing calibrations.
"""
import argparse
import pycalibtic
import pyhalco_hicann_v2 as C
from pyhalco_common import Enum
from pycake.helpers import calibtic

import pylogging
pylogging.default_config(date_format='absolute')

parser = argparse.ArgumentParser()
parser.add_argument('--wafer', required=True, type=int, help="Wafer enum")
parser.add_argument('--hicann', required=True, type=int, help="HICANNOnWafer enum")
parser.add_argument('--collection', required=True, type=str, nargs="+",
                    help="cf. calibtic's HICANNCollection",
                    choices=["Neuron", "Block", "SynapseRow", "L1Crossbar", "SynapseChainLength", "SynapseSwitch"])
parser.add_argument('--calib_path', required=True, type=str, help="path of calibration files that will be extended")
args, argv = parser.parse_known_args()

c = calibtic.Calibtic(args.calib_path, C.Wafer(args.wafer), C.HICANNOnWafer(Enum(args.hicann)))
backend = c.get_backend()
for col_name in args.collection:
    col = getattr(c.hc, "at{}Collection".format(col_name))()
    col.setDefaults()
backend.store(c.get_calibtic_name(), pycalibtic.MetaData(), c.hc)
