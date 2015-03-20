#!/usr/bin/env python
import cPickle
import argparse
import bz2
import gzip
import pylogging

from pysthal.command_line_util import init_logger
from pysthal.command_line_util import add_logger_options
from pysthal.command_line_util import folder

from pycake.calibrationrunner import CalibrationRunner

init_logger(pylogging.LogLevel.WARN, [
    ("Default", pylogging.LogLevel.INFO),
    ("halbe.fgwriter", pylogging.LogLevel.ERROR),
    ("pycake.calibrationrunner", pylogging.LogLevel.DEBUG),
    ("pycake.measurement", pylogging.LogLevel.DEBUG),
    ("pycake.analyzer", pylogging.LogLevel.TRACE),
    ("pycake.experiment", pylogging.LogLevel.DEBUG),
    ("pycake.experimentbuilder", pylogging.LogLevel.DEBUG),
    ("pycake.calibtic", pylogging.LogLevel.DEBUG),
    ("pycake.redman", pylogging.LogLevel.DEBUG),
    ("pycake.sthal", pylogging.LogLevel.DEBUG),
    ("pycake.helper.sthal", pylogging.LogLevel.INFO),
    ("sthal", pylogging.LogLevel.INFO),
    ("sthal.AnalogRecorder", pylogging.LogLevel.WARN),
    ("sthal.HICANNConfigurator.Time", pylogging.LogLevel.INFO)
    ])

parser = argparse.ArgumentParser(
        description='HICANN Calibration tool.')
add_logger_options(parser)
parser.add_argument('runner', type=folder,
        help='Pickled experiment to rerun')
args = parser.parse_args()
runner = CalibrationRunner.load(args.runner)
runner.continue_calibration()
