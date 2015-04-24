#!/usr/bin/env python

import argparse
from pprint import pprint
from defects import read_voltages

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('wafer', type=int)

    args = parser.parse_args()

    pprint(read_voltages(args.wafer))
