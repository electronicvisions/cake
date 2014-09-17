#!/bin/env python

from pycake.helpers.TraceAverager import createTraceAverager

from pysthal.util import add_default_coordinate_options
from pysthal.util import add_logger_options
from pysthal.util import init_logger

def main():
    import argparse
    init_logger('ERROR')

    parser = argparse.ArgumentParser(
            description='Determine ADC freq relative to HICANN PLL')
    add_default_coordinate_options(parser)
    add_logger_options(parser)
    args = parser.parse_args()
    averager = createTraceAverager(args.wafer, args.hicann)
    print averager.adc_freq

if __name__ == '__main__':
    main()
