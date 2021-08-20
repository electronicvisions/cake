#!/usr/bin/env python

import itertools

import pycake.helpers.plotting
import matplotlib
from pyhalco_common import iter_all, Enum
import pyhalco_hicann_v2 as C

import pycake.helpers.calibtic
import pycalibtic

import os.path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("wafer", type=int, help="wafer enum")
parser.add_argument("calib_path", type=str, help="path to calib files")
parser.add_argument('backend_type', type=str, choices=["xml", "binary"], default="xml",
                    help="Calibration backend type")
args = parser.parse_args()

calibs_to_check = [
    [pycalibtic.NeuronCalibrationParameters.Calibrations.calib.V_convoffx,
     pycalibtic.NeuronCalibrationParameters.Calibrations.calib.V_convoffi],
    [pycalibtic.NeuronCalibrationParameters.Calibrations.calib.V_syntcx,
     pycalibtic.NeuronCalibrationParameters.Calibrations.calib.V_syntci],
    [pycalibtic.NeuronCalibrationParameters.Calibrations.calib.E_synx,
     pycalibtic.NeuronCalibrationParameters.Calibrations.calib.E_syni],
    [pycalibtic.NeuronCalibrationParameters.Calibrations.calib.I_gl_slow0_fast0_bigcap1],
    [pycalibtic.NeuronCalibrationParameters.Calibrations.calib.I_pl],
    [pycalibtic.NeuronCalibrationParameters.Calibrations.calib.E_l],
    [pycalibtic.NeuronCalibrationParameters.Calibrations.calib.V_t]
    ]

wafer_c = C.Wafer(Enum(args.wafer))

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "red"])

cals = {}
for hicann_c in iter_all(C.HICANNOnWafer):
    try:
        cal = pycake.helpers.calibtic.Calibtic(args.calib_path, wafer_c, hicann_c, backend_type=args.backend_type)
        cals[hicann_c] = cal.hc.atNeuronCollection()
    except Exception as e:
        pass

figures = []

# any
data = {}
for hicann_c, nc in cals.items():
    try:
        blacklisted = 0
        for ii in range(C.NeuronOnHICANN.size):
            blacklisted += any([not nc.at(ii).exists(calib_enum) for calib_enum in itertools.chain(*calibs_to_check)])
        data[hicann_c.toEnum().value()] = blacklisted
    except IndexError as e:
        pass

names = [calib_enum.name for sublist in calibs_to_check for calib_enum in sublist]
names = "{} or {}".format(", ".join(names[:-1]), names[-1])
figures.append(pycake.helpers.plotting.get_bokeh_figure("{} not calibrated".format(names), data, cmap=cmap, add_text = True, default_fill_color = 'blue'))

# individual
for calib_enums in calibs_to_check:

    row_figures = []

    for calib_enum in calib_enums:
        data = {}
        for hicann_c, nc in cals.items():
            try:
                blacklisted = sum([not nc.at(ii).exists(calib_enum) for ii in range(C.NeuronOnHICANN.size)])
                data[hicann_c.toEnum().value()] = blacklisted
            except IndexError as e:
                pass

        row_figures.append(pycake.helpers.plotting.get_bokeh_figure(calib_enum.name, data, cmap=cmap, add_text = True, default_fill_color = 'blue'))

    figures += row_figures

pycake.helpers.plotting.store_bokeh("Calib Wafer {} Overview".format(args.wafer), figures, "calib_status_w{}.html".format(args.wafer))
