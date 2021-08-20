#!/usr/bin/env python



import argparse
from collections import defaultdict
from pyredman import load
import pylogging
from pysthal.command_line_util import init_logger

import numpy as np
from pyhalco_common import Enum
import pyhalco_hicann_v2 as C
from pycake.helpers.calibtic import Calibtic
import pycalibtic
pnc = pycalibtic.NeuronCalibration.Calibrations
odb_throw = pycalibtic.Transformation.THROW

init_logger("INFO", [("Calibtic", "ERROR")])

parser = argparse.ArgumentParser("Extract data from existing calibration")

parser.add_argument('calib_path', help="path to calibration data")
parser.add_argument('defects_path', help="path to defects data")
parser.add_argument('wafer', help="wafer number", type=int)
parser.add_argument('--hicann', help="hicann number",
                    nargs="+", type=int, required=True)
parser.add_argument('--neuron', help="neuron number",
                    nargs="+", type=int, required=True)
args = parser.parse_args()

wafer = C.Wafer(args.wafer)

vals_for_DAC = {}
vals_for_DAC[pnc.V_syntcx] = defaultdict(
    list)
vals_for_DAC[pnc.V_syntci] = defaultdict(
    list)
vals_for_DAC[pnc.I_gl_slow0_fast0_bigcap1] = defaultdict(
    list)
vals_for_DAC[pnc.I_pl] = defaultdict(
    list)

trafo_data = {}
trafo_data[pnc.E_syni] = []

hicanns = [C.HICANNOnWafer(Enum(h_enum)) for h_enum in args.hicann]
nrns = [C.NeuronOnHICANN(Enum(h_enum)) for h_enum in args.neuron]

def get_val_for_DAC(DAC,neuron_calibration, parameter, outside_domain_behaviour):
    try:
        return nc.from_dac(DAC, parameter, outside_domain_behaviour)
    except Exception as e:
        return None

for n, hicann in enumerate(hicanns):

    try:
        redman_hicann = load.HicannWithBackend(
            args.defects_path, C.HICANNGlobal(hicann, wafer),
            ignore_missing=False)
    except Exception as e:
        continue

    cal = Calibtic(args.calib_path, wafer, hicann, backend_type="binary")

    for nrn in nrns:
        print(n, hicann, nrn)
        if redman_hicann.neurons().has(nrn):
            nc = cal.get_calibration(nrn)
            try:
                # E_syni
                E_syni_data = nc.at(pnc.E_syni).getData()
                trafo_data[pnc.E_syni].append(E_syni_data)
            except Exception as e:
                pass

            for DAC in reversed(list(range(1024))):
                for parameter, vfD in list(vals_for_DAC.items()):
                    val = get_val_for_DAC(DAC, nc, parameter, odb_throw)
                    if val != None:
                        vfD[DAC].append(val)

for parameter, entries in list(vals_for_DAC.items()):
    with open('{}.dat'.format(parameter), 'w') as f:
        for DAC, vals in entries.items():
            print(np.mean(vals), np.std(vals), DAC, file=f)

for parameter, entries in list(trafo_data.items()):
    with open('{}.dat'.format(parameter), 'w') as f:
        for entry in entries:
            print(" ".join([str(v) for v in entry]), file=f)
