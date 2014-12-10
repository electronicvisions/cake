#!/usr/bin/env python
import argparse
import os
import sys
import pylogging
from pycake.calibrationrunner import CalibrationRunner, TestRunner
import pycake.config
import Coordinate

from pysthal.command_line_util import add_default_coordinate_options
from pysthal.command_line_util import add_logger_options
from pysthal.command_line_util import init_logger


def check_file(string):
    if not os.path.isfile(string):
        msg = "parameter file '%r' not found! :p" % string
        raise argparse.ArgumentTypeError(msg)
    return string


def load_config(parsed_args):
    cfg = pycake.config.Config(None, parsed_args.parameter_file)

    if parsed_args.outdir:
        backend_dir = os.path.join(parsed_args.outdir, 'backends')
        cfg.parameters['folder'] = parsed_args.outdir
        cfg.parameters['backend_c'] = backend_dir
        cfg.parameters['backend_r'] = backend_dir
    if parsed_args.hicann:
        cfg.parameters['coord_hicann'] = parsed_args.hicann
    if parsed_args.wafer:
        cfg.parameters['coord_wafer'] = parsed_args.wafer
    if parsed_args.parameter:
        for key in cfg.parameters:
            if key.startswith('run_'):
                cfg.parameters[key] = False
        cfg.parameters['run_' + parsed_args.parameter] = True
    return cfg


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
    ("sthal.HICANNConfigurator.Time", pylogging.LogLevel.DEBUG)
    ])

parser = argparse.ArgumentParser(description='HICANN Calibration tool. Takes a parameter file as input. See pycake/bin/parameters.py to see an example.')
add_default_coordinate_options(parser)
add_logger_options(parser)
parser.add_argument('parameter_file', type=check_file, help='parameterfile containing the parameters of this calibration')
parser.add_argument('--outdir', type=str, default=None, help="output folder. default is the one specified in the config file.")
parser.add_argument('--logfile', default=None,
                        help="Specify a logfile where all the logger output will be stored (any LogLevel!)")
parser.add_argument('--parameter', type=str, default=None,
                    help='Spezifiy paramter to calibrate')
args = parser.parse_args()

if args.logfile is not None:
    pylogging.log_to_file(args.logfile, pylogging.LogLevel.ALL)

config = load_config(args)

runner = CalibrationRunner(config)

if runner.config.get_run_calibration():
    runner.run_calibration()

test_runner = TestRunner(config)

if test_runner.config.get_run_test():
    test_runner.run_calibration()
