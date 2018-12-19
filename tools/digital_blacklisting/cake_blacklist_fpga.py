#!/usr/bin/env python

import os
import argparse
import subprocess
import pyhalbe.Coordinate as C

parser = argparse.ArgumentParser()
parser.add_argument("--wafer", type=int, help="Wafer enum", required=True)
parser.add_argument("--fpga", type=int, help="FPGA enum", required=True)
parser.add_argument("--seeds", type=str, help="used seeds",
                    nargs="+", required=True)
parser.add_argument("--input_backend_path", type=str,
                    help="path to comm test results", default="./")
parser.add_argument("--output_backend_path", type=str,
                    help="path where blacklisted data is stored", default="./")

args = parser.parse_args()

wafer_c = C.Wafer(args.wafer)
fpga_c = C.FPGAOnWafer(C.Enum(args.fpga))
successful = True

logbase = os.path.join(args.output_backend_path,
                       "log_wafer_" + str(args.wafer), "FPGA_" + str(args.fpga))
logfile = logbase
filecounter = 0
while(os.path.exists(logfile)):
    print ("logfile " + logfile + " already exists")
    filecounter += 1
    logfile = logbase + "_" + str(filecounter)
os.makedirs(logfile)

for c, hicann in enumerate(C.iter_all(C.HICANNOnDNC)):
    hicann_on_wafer = hicann.toHICANNOnWafer(fpga_c).toEnum().value()
    with open(os.path.join(logfile, "HICANN_" + str(hicann_on_wafer) + "_out.txt"), 'w') as fout, open(os.path.join(logfile, "HICANN_" + str(hicann_on_wafer) + "_err.txt"), 'w') as ferr:
        bl_exit = subprocess.call(["cake_digital_blacklisting", "-w", str(args.wafer), "-h", str(hicann_on_wafer), "--seeds"] + args.seeds + [
                                  "--output_backend_path", args.output_backend_path, "--input_backend_path", args.input_backend_path], stdout=fout, stderr=ferr)
        if bl_exit:
            ferr.write("Error during test of HICANN " + str(hicann_on_wafer) +
                       " on FPGA " + str(args.fpga) + " on Wafer " + str(args.wafer))
            # shows status of all tests in slurm logfile
            print ("Error during test of HICANN " + str(hicann_on_wafer) + " on FPGA " + str(args.fpga) + " on Wafer " + str(args.wafer))
            successful = False

if successful:
    print ("Test of FPGA " + str(args.fpga) + " completed successfully")
