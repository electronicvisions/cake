#!/usr/bin/env python

import pycake.helpers.plotting
import matplotlib
from pyhalco_common import iter_all
import pyhalco_hicann_v2 as C
import pyredman
import pyredman.load
import os.path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("wafer", type=int, help="wafer enum")
parser.add_argument("defects_path", type=str, help="path to redman files")
args = parser.parse_args()

wafer_with_backend = pyredman.load.WaferWithBackend(
    args.defects_path, C.Wafer(args.wafer), ignore_missing=True)

blacklisted = {}
jtag = {}

# Format [redman name, halco name (OnHICANN is added later), add text]
enum_resources = [
    ["neurons", "Neuron", "True"],
    ["drivers", "SynapseDriver", "True"],
    ["synapses", "Synapse", "False"],
    ["fgblocks", "FGBlock", "True"],
    ["vrepeaters", "VRepeater", "True"],
    ["hrepeaters", "HRepeater", "True"],
    ["synaptic_inputs", "SynapticInput", "True"],
    ["synapseswitches", "SynapseSwitch", "True"],
    ["crossbarswitches", "CrossbarSwitch", "True"],
    ["synapseswitchrows", "SynapseSwitchRow", "True"]]

non_enum_resources = [
    ["hbuses", "HLine", "True"],
    ["vbuses", "VLine", "True"],
    ["mergers0", "Merger0", "True"],
    ["mergers1", "Merger1", "True"],
    ["mergers2", "Merger2", "True"],
    ["mergers3", "Merger3", "True"],
    ["dncmergers", "DNCMerger", "True"],
    ["synapserows", "SynapseRow", "True"],
    ["analogs", "Analog", "True"],
    ["backgroundgenerators", "BackgroundGenerator", "True"]]

for res in enum_resources:
    exec (res[0] + " = {}")
for res in non_enum_resources:
    exec (res[0] + " = {}")

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

    blacklisted[hicann_id] = False

    for res in enum_resources + non_enum_resources:
        exec (
            "has_value = hicann_with_backend.{}().has_value()".format(res[0]))
        if not has_value:
            continue
        exec ("{}[hicann_id] = hicann_with_backend.{}().available()".format(
            res[0], res[0]))

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

for res in enum_resources:
    exec("max_value = C.{}OnHICANN.enum_type.end".format(res[1]))
    exec("""{}_figure = pycake.helpers.plotting.get_bokeh_figure("{}",{}, max_value, cmap, add_text = {}, default_fill_color = 'blue')""".format(
        res[0], res[0], res[0], res[2]))

for res in non_enum_resources:
    exec("max_value = C.{}OnHICANN.end".format(res[1]))
    exec("""{}_figure = pycake.helpers.plotting.get_bokeh_figure("{}",{}, max_value, cmap, add_text = {}, default_fill_color = 'blue')""".format(
        res[0], res[0], res[0], res[2]))

figures = [[blacklisted_figure], [jtag_figure]]
for res in enum_resources:
    figures.append([eval(res[0] + "_figure")])
for res in non_enum_resources:
    figures.append([eval(res[0] + "_figure")])

pycake.helpers.plotting.store_bokeh("Redman Wafer {} Overview".format(
    args.wafer), figures, "redman_w{}.html".format(args.wafer))
