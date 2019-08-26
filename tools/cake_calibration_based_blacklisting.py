#!/usr/bin/env python
"""
Blacklist neurons based on missing calibration
"""

import argparse

import pyhalco_hicann_v2 as C
from pycake.helpers.calibtic import Calibtic
from pyhalco_common import Enum
import pycalibtic
import pyredman
from pyredman.load import load

import pylogging
pylogging.default_config()

parser = argparse.ArgumentParser()
parser.add_argument('--wafer', type=int, required=True, help="Wafer enum")
parser.add_argument('--hicann', type=int, required=True, help="HICANNOnWafer enum")
parser.add_argument('--defects_path', required=True, help="path to defects files")
parser.add_argument('--calib_path', required=True, help="path to calibration files")
parser.add_argument('--parameter', required=True,
                    help="calibtic's NeuronCalibrationParameters; if neuron lacks calibration it will be blacklisted",
                    nargs="+", choices=pycalibtic.NeuronCalibrationParameters.Calibrations.calib.names.keys())
parser.add_argument('--neuron', default=range(512), type=int, nargs="+", help="NeuronOnHICANN enum to test for missing calibration")
args = parser.parse_args()

wafer_c = C.Wafer(args.wafer)
hicann_c = C.HICANNOnWafer(Enum(args.hicann))
hicann_global_c = C.HICANNGlobal(hicann_c, wafer_c)

cal = Calibtic(args.calib_path, wafer_c, hicann_c)
redman_hicann = load.HicannWithBackend(args.defects_path, hicann_global_c)

for nrn in args.neuron:

    # if the neuron is already blacklisted, we do not want to accidentally enable it
    if not redman_hicann.neurons().has(C.NeuronOnHICANN(Enum(nrn))):
        print "neuron {} already blacklisted -> skip test".format(nrn)
        continue

    disable = False
    if not cal.hc.atNeuronCollection().exists(nrn):
        print "no calibration data for neuron {}".format(nrn)
        disable = True
    else:
        for param in args.parameter:
            enum = getattr(pycalibtic.NeuronCalibrationParameters.Calibrations.calib, param)
            has_param = cal.hc.atNeuronCollection().at(nrn).exists(enum)
            if not has_param:
                disable = True
                print "{} missing for neuron {}".format(param, nrn)
                # one calibration paramter missing is enough to disable
                break

    if disable:
        print "disabling {} neuron {}".format(C.short_format(hicann_global_c), nrn)
        redman_hicann.neurons().disable(C.NeuronOnHICANN(Enum(nrn)), pyredman.switch_mode.NONTHROW)
    else:
        redman_hicann.neurons().enable(C.NeuronOnHICANN(Enum(nrn)), pyredman.switch_mode.NONTHROW)

redman_hicann.save()
