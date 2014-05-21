#!/usr/bin/env python
import cPickle
import argparse
import bz2
import pylogging

pylogging.default_config(date_format='absolute')
pylogging.set_loglevel(pylogging.get("Default"),                    pylogging.LogLevel.ERROR)
pylogging.set_loglevel(pylogging.get("pycake.calibrationrunner"),   pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.measurement"),         pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.experiment"),          pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.experimentbuilder"),   pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("pycake.calibtic"),            pylogging.LogLevel.DEBUG )
pylogging.set_loglevel(pylogging.get("sthal"),                      pylogging.LogLevel.INFO )
pylogging.set_loglevel(pylogging.get("sthal.AnalogRecorder"),       pylogging.LogLevel.WARN)

parser = argparse.ArgumentParser(
        description='HICANN Calibration tool.') 
parser.add_argument('runner_file', type=argparse.FileType('r'),
        help='Pickled experiment to rerun')
args = parser.parse_args()

runner_filename = args.runner_file.name
if runner_filename.endswith('.bz2'):
    with bz2.BZ2File(runner_filename) as infile:
        runner = cPickle.load(infile)
else:
    runner = cPickle.load(args.runner_file)

runner.continue_calibration()
