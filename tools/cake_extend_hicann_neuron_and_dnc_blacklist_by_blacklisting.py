#!/usr/bin/env python

from pyhalco_hicann_v2 import Wafer, FPGAGlobal, FPGAOnWafer, HICANNGlobal, short_format
from pyhalco_common import iter_all
from pyredman.load import load
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('wafer', type=int, help="Wafer for which to create blacklisting for")
parser.add_argument('input_dir', help="where to read redman files from")
parser.add_argument('output_dir', help="where to store redman files")
args = parser.parse_args()

wafer_c = Wafer(args.wafer)

for fpga in iter_all(FPGAOnWafer):
    fpga_backend = load.FpgaWithBackend(args.input_dir, FPGAGlobal(fpga, wafer_c))
    for hicann in fpga.toHICANNOnWafer():
        hslink = hicann.toHighspeedLinkOnDNC()
        # check if highspeed is blacklisted
        if not fpga_backend.hslinks().has(hslink):
            # blacklist neurons and DNCMerger
            hicann_backend = load.HicannWithBackend(args.output_dir, HICANNGlobal(hicann, wafer_c))
            hicann_backend.neurons().disable_all()
            hicann_backend.dncmergers().disable_all()
            hicann_backend.save()
            print ("disable neurons and dncmerger on {} {}"
                   .format(short_format(wafer_c),short_format(hicann)))
