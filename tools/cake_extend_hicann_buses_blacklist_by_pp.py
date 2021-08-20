#!/usr/bin/env python

import argparse
from collections import defaultdict

from pyhalco_hicann_v2 import Wafer, HICANNGlobal, HICANNOnWafer, HLineOnHICANN, VLineOnHICANN
from pyhalco_hicann_v2 import short_format
from pyhalco_common import Enum, iter_all
from pyredman.load import load

# connected via postprocessing to reticle without fpga
connected_via_postprocessing = defaultdict(list)

for h_left in [HICANNOnWafer(Enum(e)) for e in (0, 12, 24, 44, 320, 340, 360, 372)]:
    connected_via_postprocessing[h_left].append("west")

for h_right in [HICANNOnWafer(Enum(e)) for e in (11, 23, 43, 63, 339, 359, 371, 383)]:
    connected_via_postprocessing[h_right].append("east")

for h_top in [HICANNOnWafer(Enum(e)) for e in (24, 25, 26, 27, 64, 65, 66, 67, 40, 41, 42, 43, 88, 89, 90, 91)]:
    connected_via_postprocessing[h_top].append("north")

for h_bottom in [HICANNOnWafer(Enum(e)) for e in (292, 293, 294, 295, 340, 341, 342, 343, 316, 317, 318, 319, 356, 357, 358, 359)]:
    connected_via_postprocessing[h_bottom].append("south")

# derive to be blacklisted buses
blacklist_hbuses = defaultdict(list)
blacklist_vbuses = defaultdict(list)

for hicann, direction in connected_via_postprocessing.items():
    for v_line in iter_all(VLineOnHICANN):
        if ("north" in direction and v_line.toVRepeaterOnHICANN().isBottom()) or \
           ("south" in direction and v_line.toVRepeaterOnHICANN().isTop()):
            blacklist_vbuses[hicann].append(v_line)

    for h_line in iter_all(HLineOnHICANN):
        if ("east" in direction and h_line.toHRepeaterOnHICANN().isLeft()) or \
           ("west" in direction and h_line.toHRepeaterOnHICANN().isRight()):
            blacklist_hbuses[hicann].append(h_line)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('wafer', type=int, help="Wafer for which to create blacklisting for")
    parser.add_argument('outdir', help="where to store redman files")
    args = parser.parse_args()

    wafer_c = Wafer(args.wafer)

    for hicann, hbuses in blacklist_hbuses.items():
        redman_hicann = load.HicannWithBackend(args.outdir, HICANNGlobal(hicann, wafer_c))
        for hb in hbuses:
            if redman_hicann.hbuses().has(hb):
                redman_hicann.hbuses().disable(hb)
                print(("disable {} on {}".format(hb, short_format(hicann))))
            else:
                print(("{} on {} already disabled -> skipped".format(hb, short_format(hicann))))
        redman_hicann.save()

    for hicann, vbuses in blacklist_vbuses.items():
        redman_hicann = load.HicannWithBackend(args.outdir, HICANNGlobal(hicann, wafer_c))
        for vb in vbuses:
            if redman_hicann.vbuses().has(vb):
                redman_hicann.vbuses().disable(vb)
                print(("disable {} on {}".format(vb, short_format(hicann))))
            else:
                print(("{} on {} already disabled -> skipped".format(vb, short_format(hicann))))
        redman_hicann.save()

