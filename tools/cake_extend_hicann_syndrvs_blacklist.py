#!/usr/bin/env python

import json
import argparse
import pyredman as redman
from pyhalco_hicann_v2 import Wafer, HICANNGlobal, HICANNOnWafer, SynapseDriverOnHICANN
from pyhalco_common import Enum
def init_backend(fname):
    lib = redman.loadLibrary(fname)
    backend = redman.loadBackend(lib)
    if not backend:
        raise Exception('unable to load %s' % fname)
    return backend

redman_backend = init_backend('libredman_xml.so')


parser = argparse.ArgumentParser()
parser.add_argument('--wafer', required=True, type=int)
parser.add_argument('--hicann', nargs="+", required=True, type=int)
parser.add_argument('--defect_files', nargs="+", type=argparse.FileType('r'), default=[])
parser.add_argument('--defects_path', required=True)
parser.add_argument('--syndrv', nargs="+", default=[], type=int)
parser.add_argument('--hicann_version', default="v4")
args, argv = parser.parse_known_args()

redman_backend.config('path', args.defects_path)
redman_backend.init()

if args.hicann_version == "v4":
    unavailable_in_hicann_version = [110,111,112,113]
elif args.hicann_version == "v2":
    unavailable_in_hicann_version = []
else:
    raise RuntimeError("unknown HICANN version {}".format(args.hicann_version))

for hicann in args.hicann:
    print("w{}h{}".format(args.wafer, hicann))
    redman_hicann = redman.HicannWithBackend(redman_backend, HICANNGlobal(HICANNOnWafer(Enum(hicann)), Wafer(args.wafer)))
    for drv in unavailable_in_hicann_version:
        drv_c = SynapseDriverOnHICANN(Enum(drv))
        if redman_hicann.drivers().has(drv_c):
            print("\tdisabling {}".format(drv_c))
            redman_hicann.drivers().disable(drv_c)
        else:
            print("\t{} already disabled".format(drv_c))
    redman_hicann.save()

for df_file in args.defect_files:
    try:
        df = json.load(df_file)
        if df['hicann'] in args.hicann:
            print("w{}h{}".format(args.wafer, df['hicann']))
            redman_hicann = redman.HicannWithBackend(redman_backend, HICANNGlobal(HICANNOnWafer(Enum(df['hicann'])), Wafer(args.wafer)))
            for drv in df['bad_drivers']:
                drv_c = SynapseDriverOnHICANN(Enum(drv))
                if redman_hicann.drivers().has(drv_c):
                    print("\tdisabling {}".format(drv_c))
                    redman_hicann.drivers().disable(drv_c)
                else:
                    print("\t{} already disabled".format(drv_c))
            redman_hicann.save()
    except Exception as e:
        print(e)
        continue

for drv in args.syndrv:
    for hicann in args.hicann:
        redman_hicann = redman.HicannWithBackend(redman_backend, HICANNGlobal(HICANNOnWafer(Enum(hicann)), Wafer(args.wafer)))
        drv_c = SynapseDriverOnHICANN(Enum(drv))
        if redman_hicann.drivers().has(drv_c):
            print("\tdisabling {}".format(drv_c))
            redman_hicann.drivers().disable(drv_c)
        else:
            print("\t{} already disabled".format(drv_c))
        redman_hicann.save()
