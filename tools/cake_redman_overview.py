#!/usr/bin/env python

import pycake.helpers.plotting
import matplotlib
from pyhalco_common import iter_all
import pyhalco_hicann_v2 as C
from pyredman import load
import os.path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("wafer", type=int, help="wafer enum")
parser.add_argument("defects_path", type=str, help="path to redman files")
args = parser.parse_args()

wafer_with_backend = load.WaferWithBackend(
    args.defects_path, C.Wafer(args.wafer), ignore_missing=True)

blacklisted = {}
jtag = {}

# Format [redman name, add text]
resources = [
    ["neurons", True],
    ["drivers", True],
    ["synapses", False],
    ["fgblocks", True],
    ["vrepeaters", True],
    ["hrepeaters", True],
    ["synaptic_inputs", True],
    ["synapseswitches", True],
    ["crossbarswitches", True],
    ["synapseswitchrows", True],
    ["hbuses", True],
    ["vbuses", True],
    ["mergers0", True],
    ["mergers1", True],
    ["mergers2", True],
    ["mergers3", True],
    ["dncmergers", True],
    ["synapserows", True],
    ["analogs", True],
    ["backgroundgenerators", True]]

# store blacklisting data for each component and each HICANN
components = {}
for res in resources:
    components[res[0]] = {}

for hicann in iter_all(C.HICANNOnWafer):
    hicann_with_backend = wafer_with_backend.get(hicann)
    fpga_with_backend = wafer_with_backend.get(hicann.toFPGAOnWafer())
    hicann_id = hicann.toEnum().value()

    # fill blacklisted HICANNs and continue if blacklisted (since no additional information available)
    blacklisted[hicann_id] = not wafer_with_backend.hicanns().has(hicann)
    if (not wafer_with_backend.hicanns().has(hicann)):
        continue
    jtag[hicann_id] = not fpga_with_backend.hslinks().has(
        hicann.toHighspeedLinkOnDNC())
    # check if HICANN file exists for not blacklisted HICANNS
    if not os.path.exists("{}/hicann-Wafer({})-Enum({}).xml".format(args.defects_path, args.wafer, hicann_id)):
        continue

    # check resources
    for res in resources:
        has_value = getattr(hicann_with_backend, res[0])().has_value()
        if not has_value:
            # skip not tested components
            continue
        # store number of available components
        components[res[0]][hicann_id] = getattr(
            hicann_with_backend, res[0])().available()

# generate JTAG with green yellow
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["green", "yellow"])
max_value = 1
jtag_figure = pycake.helpers.plotting.get_bokeh_figure(
    "JTAG", jtag, max_value, cmap, default_fill_color='blue')
# generate all other colormaps
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [
                                                           "green", "red"])
max_value = 1
blacklisted_figure = pycake.helpers.plotting.get_bokeh_figure(
    "blacklisted", blacklisted, max_value, cmap, default_fill_color='blue')
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [
                                                           "red", "green"])

figures = [[blacklisted_figure], [jtag_figure]]
for res in resources:
    # coordinate is either enum or non enum type
    try:
        max_value = getattr(hicann_with_backend,
                            res[0])().resource.enum_type.end
    except AttributeError:
        max_value = getattr(hicann_with_backend, res[0])().resource.end
    figures.append(pycake.helpers.plotting.get_bokeh_figure(
        res[0], components[res[0]], max_value, cmap, add_text=res[1], default_fill_color='blue'))

pycake.helpers.plotting.store_bokeh("Redman Wafer {} Overview".format(
    args.wafer), figures, "redman_w{}.html".format(args.wafer))
