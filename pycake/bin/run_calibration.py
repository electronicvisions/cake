#!/usr/bin/env python
import pickle
import argparse
import os
import sys
import pylogging
from pycake.calibrationrunner import CalibrationRunner, TestRunner


def check_file(string):
    if not os.path.isfile(string):
        msg = "parameter file '%r' not found! :p" % string
        raise argparse.ArgumentTypeError(msg)
    return string

parser = argparse.ArgumentParser(description='HICANN Calibration tool. Takes a parameter file as input. See pycake/bin/parameters.py to see an example.')
parser.add_argument('parameter_file', type=check_file, help='parameterfile containing the parameters of this calibration')
parser.add_argument('--logfile', default=None,
                        help="Specify a logfile where all the logger output will be stored (any LogLevel!)")
args = parser.parse_args()

logfile = args.logfile

config_filename = args.parameter_file

pylogging.default_config(date_format='absolute')
pylogging.set_loglevel(pylogging.get("Default"),                    pylogging.LogLevel.INFO)
pylogging.set_loglevel(pylogging.get("pycake.calibrationrunner"),   pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.measurement"),         pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.analyzer"),            pylogging.LogLevel.TRACE )
pylogging.set_loglevel(pylogging.get("pycake.experiment"),          pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.experimentbuilder"),   pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.calibtic"),            pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.redman"),              pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.sthal"),            pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.helper.sthal"),        pylogging.LogLevel.INFO )
pylogging.set_loglevel(pylogging.get("sthal"),                      pylogging.LogLevel.INFO )
pylogging.set_loglevel(pylogging.get("sthal.AnalogRecorder"),       pylogging.LogLevel.WARN)

if logfile is not None:
    pylogging.log_to_file(logfile, pylogging.LogLevel.ALL)

runner = CalibrationRunner(config_filename)
if runner.config.get_run_calibration():
    runner.run_calibration()

test_runner = TestRunner(config_filename)

pylogging.reset()
pylogging.default_config()
pylogging.set_loglevel(pylogging.get("Default"),                    pylogging.LogLevel.INFO)
pylogging.set_loglevel(pylogging.get("pycake.testrunner"),          pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.measurement"),         pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.experiment"),          pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.experimentbuilder"),   pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.calibtic"),            pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.redman"),              pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.sthal"),            pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.helper.sthal"),        pylogging.LogLevel.INFO )
pylogging.set_loglevel(pylogging.get("sthal"),                      pylogging.LogLevel.INFO )
pylogging.set_loglevel(pylogging.get("sthal.AnalogRecorder"),       pylogging.LogLevel.WARN)

if logfile is not None:
    pylogging.log_to_file(logfile, pylogging.LogLevel.ALL)

if test_runner.config.get_run_test():
    test_runner.run_calibration()
