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
    ("pycake.calibrationunit", pylogging.LogLevel.DEBUG),
    ("pycake.measurement", pylogging.LogLevel.DEBUG),
    ("pycake.analyzer", pylogging.LogLevel.TRACE),
    ("pycake.experiment", pylogging.LogLevel.DEBUG),
    ("pycake.experimentbuilder", pylogging.LogLevel.DEBUG),
    ("pycake.calibtic", pylogging.LogLevel.DEBUG),
    ("pycake.redman", pylogging.LogLevel.DEBUG),
    ("pycake.helper.sthal", pylogging.LogLevel.INFO),
    ("pycake.helper.simsthal", pylogging.LogLevel.INFO),
    ("sthal", pylogging.LogLevel.INFO),
    ("sthal.AnalogRecorder", pylogging.LogLevel.WARN),
    ("sthal.HICANNConfigurator.Time", pylogging.LogLevel.INFO),
    ("progress", pylogging.LogLevel.DEBUG),
    ("progress.sthal", pylogging.LogLevel.INFO),
    ])

parser = argparse.ArgumentParser(
        description='HICANN Calibration tool.')
add_logger_options(parser)
parser.add_argument('runner', type=folder,
        help='Pickled experiment to rerun')
parser.add_argument('--parameter', type=str, default=None, action='append',
                    help='Resumes only the specified calibrations')
args = parser.parse_args()

if args.logfile is not None:
    progress = pylogging.get('progress')
    pylogging.append_to_cout(progress)
    progress.warn("Write logs to {}".format(args.logfile))

runner = CalibrationRunner.load(args.runner)
runner.continue_calibration(args.parameter)
runner.finalize()
