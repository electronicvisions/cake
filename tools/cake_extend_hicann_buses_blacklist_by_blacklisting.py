#!/usr/bin/env python

import argparse
from pyhalco_hicann_v2 import Wafer, HICANNGlobal, HICANNOnWafer, HRepeaterOnWafer
from pyhalco_hicann_v2 import HRepeaterOnHICANN, VRepeaterOnWafer, VRepeaterOnHICANN
from pyhalco_common import iter_all

from pyredman.load import load

def disable_by_neighbor_hrepeater(hr, outdir):
    hline_on_neighbor = hr.toHLineOnWafer()[1]
    if (hline_on_neighbor):
        redman_hicann_output = load.HicannWithBackend(outdir,
                                                      HICANNGlobal(hline_on_neighbor.toHICANNOnWafer(), wafer_c))
        hline = hline_on_neighbor.toHLineOnHICANN()
        redman_hicann_output.hbuses().disable(hline)
        redman_hicann_output.save()
        print ("disable hline {}".format(hline))

def disable_by_neighbor_vrepeater(vr, outdir):
    vline_on_neighbor = vr.toVLineOnWafer()[1]
    if (vline_on_neighbor):
        redman_hicann_output = load.HicannWithBackend(outdir,
                                                      HICANNGlobal(vline_on_neighbor.toHICANNOnWafer(), wafer_c))
        vline = vline_on_neighbor.toVLineOnHICANN()
        redman_hicann_output.vbuses().disable(vline)
        redman_hicann_output.save()
        print ("disable vline {}".format(vline))

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
                try:
                    hr_on_wafer = HRepeaterOnWafer(hr,hicann)
                    disable_by_neighbor_hrepeater(hr_on_wafer, args.output_dir)
                except Exception as e:
                    print e

        for vr in iter_all(VRepeaterOnHICANN):
            if not redman_wafer_input.has(hicann) or not redman_hicann_input.vrepeaters().has(vr):
                try:
                    vr_on_wafer = VRepeaterOnWafer(vr,hicann)
                    disable_by_neighbor_vrepeater(vr_on_wafer, args.output_dir)
                except Exception as e:
                    print e
