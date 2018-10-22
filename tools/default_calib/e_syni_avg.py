#!/usr/bin/env python

from __future__ import print_function

import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser("Average E_syni transformation parameters")
parser.add_argument('file', type=argparse.FileType('r'),
                    help="file containing E_syni transformation parameters per line")
parser.add_argument('--outputfilename', default=None,
                    help="store averaged transformation parameters to file"
                    "printed to stdout if not given")
args = parser.parse_args()

data = np.loadtxt(args.file.name)

f = open(args.outputfilename, 'w') if args.outputfilename else sys.stdout
print("{},\n{}".format(np.mean(data[:, 0]), np.mean(data[:, 1])), file=f)
