#!/usr/bin/env python

import argparse
from pyhalco_hicann_v2 import Wafer, HICANNGlobal, HICANNOnWafer, RepeaterBlockOnHICANN
from pyhalco_hicann_v2 import HRepeaterOnHICANN, VRepeaterOnHICANN
from pyhalco_hicann_v2 import short_format
from pyhalco_common import iter_all

from pyredman.load import load


def blacklist_repeaters_on_block(redman, repeater_block):
    for hr in iter_all(HRepeaterOnHICANN):
        if hr.toRepeaterBlockOnHICANN() == repeater_block and redman.hrepeaters().has(hr):
            redman.hrepeaters().disable(hr)
    for vr in iter_all(VRepeaterOnHICANN):
        if vr.toRepeaterBlockOnHICANN() == repeater_block and redman.vrepeaters().has(vr):
            redman.vrepeaters().disable(vr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Blacklist all repeaters on repeater blocks with more than one defect repeater.')
    parser.add_argument('--wafer', type=int, required=True,
                        help="Wafer for which to create blacklisting for")
    parser.add_argument('--input_dir', type=str, required=True, help="Where to read redman files from")
    parser.add_argument('--output_dir', type=str, required=True, help="Where to store redman files. "
                        "If file already exists, it gets adapted")
    args = parser.parse_args()

    wafer_c = Wafer(args.wafer)

    redman_wafer_input = load.WaferWithBackend(args.input_dir, wafer_c)

    for hicann in iter_all(HICANNOnWafer):
        hicann_global = HICANNGlobal(hicann, wafer_c)
        redman_hicann_input = load.HicannWithBackend(
            args.input_dir, hicann_global)
        hrepeaters = redman_hicann_input.hrepeaters()
        vrepeaters = redman_hicann_input.vrepeaters()
        # skip hicanns with no blacklisted repeaters to reduce runtime
        no_hrepeater_defects = hrepeaters.available() == HRepeaterOnHICANN.size
        no_vrepeater_defects = vrepeaters.available() == VRepeaterOnHICANN.size
        if (no_hrepeater_defects and no_vrepeater_defects):
            continue
        # count defects on each repeater block
        defects_on_repeater_block = {
            rb: 0 for rb in iter_all(RepeaterBlockOnHICANN)}
        for hr in iter_all(HRepeaterOnHICANN):
            if not hrepeaters.has(hr):
                defects_on_repeater_block[hr.toRepeaterBlockOnHICANN()] += 1
        for vr in iter_all(VRepeaterOnHICANN):
            if not vrepeaters.has(vr):
                defects_on_repeater_block[vr.toRepeaterBlockOnHICANN()] += 1
        # blacklist all repeaters of repeater blocks with more than one defect
        # repeater
        redman_hicann_output = load.HicannWithBackend(
            args.output_dir, hicann_global)
        blacklisted = False
        for rb in iter_all(RepeaterBlockOnHICANN):
            if defects_on_repeater_block[rb] > 1:
                print ("disable all repeaters on {} on {}".format(
                    short_format(rb), short_format(hicann_global)))
                blacklist_repeaters_on_block(redman_hicann_output, rb)
                blacklisted = True
        if blacklisted:
            redman_hicann_output.save()
