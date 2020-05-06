#!/usr/bin/env python

"""
Blacklists all HICANNs with at least one blacklisted floating gate block
"""
from pyhalco_hicann_v2 import Wafer, HICANNOnWafer, HICANNGlobal
from pyhalco_hicann_v2 import FGBlockOnHICANN, short_format
from pyhalco_common import Enum
from pyredman.load import load
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--wafer', type=int, required=True,
                    help="Wafer for which to create blacklisting for")
parser.add_argument('--input_dir', type=str, required=True, help="where to read redman files from")
parser.add_argument('--output_dir', type=str, required=True, help="where to store redman files. "
                    "If file already exists, it gets adapted")
parser.add_argument('--hicanns', nargs="+", type=int, default=range(HICANNOnWafer.enum_type.end),
                    help="HICANNs for which to create blacklisting for")
args = parser.parse_args()

wafer_c = Wafer(args.wafer)
wafer_output_backend = load.WaferWithBackend(args.output_dir, wafer_c)

for hicann in args.hicanns:
    hicann_c = HICANNOnWafer(Enum(hicann))
    # skip already disabled HICANNs, test unnecessary
    if not wafer_output_backend.hicanns().has(hicann_c):
        print ("{} on {} already disabled -> skip test"
               .format(short_format(hicann_c), short_format(wafer_c)))
        continue
    hicann_input_backend = load.HicannWithBackend(
        args.input_dir, HICANNGlobal(hicann_c, wafer_c))
    if hicann_input_backend.fgblocks().available() != FGBlockOnHICANN.size:
        wafer_output_backend.hicanns().disable(hicann_c)
        print ("disable {} on {}"
               .format(short_format(hicann_c), short_format(wafer_c)))
wafer_output_backend.save()
