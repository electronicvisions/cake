#!/usr/bin/env python
import pickle
import argparse
import os
import pylogging
from pycake.calibrationrunner import CalibrationRunner

def check_file(string):
    if not os.path.isfile(string):
        msg = "parameter file '%r' not found! :p" % string
        raise argparse.ArgumentTypeError(msg)
    return string

parser = argparse.ArgumentParser(description='HICANN Calibration tool. Takes a parameter file as input. See pycake/bin/parameters.py to see an example.')
parser.add_argument('parameter_file', type=check_file, help='')
args = parser.parse_args()


config_filename = args.parameter_file

runner = CalibrationRunner(config_filename)

pylogging.default_config()
pylogging.set_loglevel(pylogging.get("Default"),                    pylogging.LogLevel.ERROR)
pylogging.set_loglevel(pylogging.get("pycake.calibrationrunner"),   pylogging.LogLevel.TRACE)
pylogging.set_loglevel(pylogging.get("pycake.measurement"),         pylogging.LogLevel.ALL)
pylogging.set_loglevel(pylogging.get("pycake.experiment"),          pylogging.LogLevel.ALL)
pylogging.set_loglevel(pylogging.get("pycake.experimentbuilder"),    pylogging.LogLevel.TRACE)

runner.run_calibration()

pickle.dump(runner, open('/home/np001/temp/restructure/runner.p', 'wb'))




test_runner = CalibrationRunner(config_filename, test=True)

pylogging.reset()
pylogging.default_config()
pylogging.set_loglevel(pylogging.get("Default"),                    pylogging.LogLevel.ERROR)
pylogging.set_loglevel(pylogging.get("pycake.calibrationrunner"),   pylogging.LogLevel.TRACE)
pylogging.set_loglevel(pylogging.get("pycake.measurement"),         pylogging.LogLevel.ALL)
pylogging.set_loglevel(pylogging.get("pycake.experiment"),          pylogging.LogLevel.ALL)
pylogging.set_loglevel(pylogging.get("pycake.experimentbuilder"),    pylogging.LogLevel.TRACE)

pickle.dump(runner, open('/home/np001/temp/restructure/runner_test.p', 'wb'))

test_runner.run_calibration()
