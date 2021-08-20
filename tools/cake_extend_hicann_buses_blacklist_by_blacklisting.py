#!/usr/bin/env python

import argparse
from pyhalco_hicann_v2 import Wafer, HICANNGlobal, HICANNOnWafer, HRepeaterOnWafer
from pyhalco_hicann_v2 import HRepeaterOnHICANN, VRepeaterOnWafer, VRepeaterOnHICANN
from pyhalco_hicann_v2 import short_format
from pyhalco_common import iter_all

from pyredman.load import load

def disable_hline(hl, outdir, wafer):
    hicann_global = HICANNGlobal(hl.toHICANNOnWafer(), wafer)
    redman_hicann_output = load.HicannWithBackend(outdir, hicann_global)
    hline = hl.toHLineOnHICANN()
    redman_hicann_output.hbuses().disable(hline)
    redman_hicann_output.save()
    print(("disable {} on {}".format(hline, short_format(hicann_global))))

def disable_vline(vl, outdir, wafer):
    hicann_global = HICANNGlobal(vl.toHICANNOnWafer(), wafer)
    redman_hicann_output = load.HicannWithBackend(outdir, hicann_global)
    vline = vl.toVLineOnHICANN()
    redman_hicann_output.vbuses().disable(vline)
    redman_hicann_output.save()
    print(("disable {} on {}".format(vline, short_format(hicann_global))))

def disable_by_local_hrepeater(hr, outdir, wafer):
    local_hline = hr.toHLineOnWafer()[0]
    disable_hline(local_hline, outdir, wafer)

def disable_by_neighbor_hrepeater(hr, outdir, wafer):
    neighboring_hline = hr.toHLineOnWafer()[1]
    # check if neighboring hline exists
    if (neighboring_hline):
        disable_hline(neighboring_hline, outdir, wafer)

def disable_by_local_vrepeater(vr, outdir, wafer):
    local_vline = vr.toVLineOnWafer()[0]
    disable_vline(local_vline, outdir, wafer)

def disable_by_neighbor_vrepeater(vr, outdir, wafer):
    neighboring_vline = vr.toVLineOnWafer()[1]
    # check if neighboring vline exists
    if (neighboring_vline):
        disable_vline(neighboring_vline, outdir, wafer)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('wafer', type=int, help="Wafer for which to create blacklisting for")
    parser.add_argument('input_dir', help="where to read redman files from")
    parser.add_argument('output_dir', help="where to store redman files")
    args = parser.parse_args()

    wafer_c = Wafer(args.wafer)

    redman_wafer_input = load.WaferWithBackend(args.input_dir, wafer_c)

    for hicann in iter_all(HICANNOnWafer):
        redman_hicann_input = load.HicannWithBackend(args.input_dir, HICANNGlobal(hicann, wafer_c))
        for hr in iter_all(HRepeaterOnHICANN):
            if not redman_wafer_input.has(hicann) or not redman_hicann_input.hrepeaters().has(hr):
                hr_on_wafer = HRepeaterOnWafer(hr,hicann)
                try:
                    disable_by_local_hrepeater(hr_on_wafer, args.output_dir, wafer_c)
                except RuntimeError as e:
                    print(e)
                try:
                    disable_by_neighbor_hrepeater(hr_on_wafer, args.output_dir, wafer_c)
                except RuntimeError as e:
                    print(e)

        for vr in iter_all(VRepeaterOnHICANN):
            if not redman_wafer_input.has(hicann) or not redman_hicann_input.vrepeaters().has(vr):
                vr_on_wafer = VRepeaterOnWafer(vr,hicann)
                try:
                    disable_by_local_vrepeater(vr_on_wafer, args.output_dir, wafer_c)
                except RuntimeError as e:
                    print(e)
                try:
                    disable_by_neighbor_vrepeater(vr_on_wafer, args.output_dir, wafer_c)
                except RuntimeError as e:
                    print(e)
