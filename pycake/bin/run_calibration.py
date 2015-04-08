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

class DictionaryAction(argparse.Action):
    """Converts values to the given time and collects them in a dictionary"""
    TYPES = {
        'float': float,
        'int': int,
        'string': str,
    }
    METAVAR = ('<key>', '<type>', '<value>')

    def __init__(self, option_strings, dest, **kwargs):
        argparse.Action.__init__(
            self, option_strings, dest, nargs=3, metavar=self.METAVAR,
            default={}, **kwargs)

    def __call__(self, parser, namespace, arg_value, option_string):
        target = getattr(namespace, self.dest)
        if target is None:
            target = {}

        try:
            key, value_type, value = arg_value
            target[key] = self.TYPES[value_type](value)
        except KeyError as e:
            raise argparse.ArgumentError(
                self,
                "Unknown value type '{}', known types are: {}".format(
                    value_type, ", ".join(self.TYPES.keys())))
        except ValueError as e:
            raise argparse.ArgumentError(
                self,
                "Could not convert '{}' to {}: {}".format(
                    value, value_type, e))
        except Exception as e:
            raise argparse.ArgumentError(
                self,
                "Strange Error parsing '{}': {}".format(
                    option_string, e))
        setattr(namespace, self.dest, target)


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
        remove = [key for key in cfg.parameters if key.startswith('run_')]
        for key in remove:
            del cfg.parameters[key]
        cfg.parameters['parameter_order'] = parsed_args.parameter
    cfg.parameters.update(parsed_args.overwrite)
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
    ("sthal.HICANNConfigurator.Time", pylogging.LogLevel.INFO)
    ])

parser = argparse.ArgumentParser(description='HICANN Calibration tool. Takes a parameter file as input. See pycake/bin/parameters.py to see an example.')
add_default_coordinate_options(parser)
add_logger_options(parser)
parser.add_argument('parameter_file', type=check_file,
                    help="parameterfile containing the parameters of this "
                         "calibration")
parser.add_argument('--outdir', type=str, default=None,
                    help="output folder. default is the one specified in the "
                         "config file.")
parser.add_argument('--logfile', default=None,
                    help="Specify a logfile where all the logger output will "
                         "be stored")
parser.add_argument('--parameter', type=str, default=None, action='append',
                    help='Runs the specified calibrations in the given order')
parser.add_argument('--overwrite', required=False, action=DictionaryAction,
                    help="Overwrites values loaded from configuration file, e.g. --overwrite PLL float 125e6")
args = parser.parse_args()

if args.logfile is not None:
    pylogging.append_to_file(args.logfile, pylogging.get_root())

config = load_config(args)

if config.get_run_calibration():
    runner = CalibrationRunner(config)
    runner.run_calibration()

if config.get_run_test():
    test_runner = TestRunner(config)
    test_runner.run_calibration()
