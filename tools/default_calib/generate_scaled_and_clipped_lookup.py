#!/usr/bin/env python

from __future__ import print_function

import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser("Generate clipped and scaled lookup tables")
parser.add_argument('file', type=argparse.FileType('r'),
                    help="file containing an averaged lookup table")

parser.add_argument('--scale_factor', default=1, type=float,
                    help="factor with which tau_syn_I values are scaled")

parser.add_argument('--min_max_dac', default=[0, 1023], type=int, nargs=2,
                    help="DAC value at which the lookup table starts")

parser.add_argument('--outputfilename', default=None,
                    help="store clipped (and scaled) lookup table to file"
                    "printed to stdout if not given")

parser.add_argument('--prohibit_reordering', default=False, action="store_true",
                    help="if set abort, otherwise reorder the not ordered values")

parser.add_argument('--increasing', default=False, action="store_true",
                    help="values are increasing")

args = parser.parse_args()

data = np.loadtxt(args.file.name)

# check for ordering and reorder if requested
ordered = np.all(np.diff(data[:,0]) >= 0) if args.increasing else np.all(np.diff(data[:,0]) <= 0)

if not ordered and args.prohibit_reordering:
    raise RuntimeError("Data not ordered!")

if not ordered:
    for n in range(len(data)-1):
        d0 = data[:,0][n]
        d1 = data[:,0][n+1]

        ordered = d1 >= d0 if args.increasing else d1 <= d0
        if not ordered:
            data[:,0][n+1] = d0

# recheck ordering
ordered = np.all(np.diff(data[:,0]) >= 0) if args.increasing else np.all(np.diff(data[:,0]) <= 0)
if not ordered:
    raise RuntimeError("Order fudging failed!")

f = open(args.outputfilename, 'w') if args.outputfilename else sys.stdout
data_filtered = data[np.logical_and(args.min_max_dac[0] <= data[:,2], data[:,2] <= args.min_max_dac[1])]
print(",\n".join(str(d*args.scale_factor) for d in data_filtered[:,0]),
      file=f)
