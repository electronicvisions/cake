#!/usr/bin/env python

import os
import errno
import argparse
import cake_calib_status

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

parser = argparse.ArgumentParser()
parser.add_argument('input', help="input directory")
parser.add_argument('output', help="output directory (will be be created)")

args = parser.parse_args()

mkdir_p(args.output)

cake_calib_status.copy_complete_calibs(args.input,
                                       args.output)
