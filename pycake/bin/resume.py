#!/usr/bin/env python
import cPickle
import argparse
import bz2
import gzip
import pylogging

from pycake.calibrationrunner import CalibrationRunner
from pycake.helpers.init_logging import init_cake_logging
from pysthal.command_line_util import add_logger_options

init_cake_logging()

parser = argparse.ArgumentParser(
        description='HICANN Calibration tool.')
add_logger_options(parser)
parser.add_argument('runner', type=folder,
        help='Pickled experiment to rerun')
parser.add_argument('--parameter', type=str, default=None, action='append',
                    help='Resumes only the specified calibrations')
args = parser.parse_args()

runner = CalibrationRunner.load(args.runner)
runner.continue_calibration(args.parameter)
runner.finalize()
