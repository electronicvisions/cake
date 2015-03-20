#!/usr/bin/env python
import argparse
import os
import shutil

import pylogging

from pycake.helpers.misc import load_pickled_file
from pycake.calibrationrunner import CalibrationRunner
from pycake.calibrationrunner import CalibrationStep
from pysthal.command_line_util import add_logger_options
from pysthal.command_line_util import init_logger

logger = pylogging.get('main')

def check_file(string):
    if not os.path.isfile(string):
        msg = "parameter file '%r' not found! :p" % string
        raise argparse.ArgumentTypeError(msg)
    return string

def convert_old_runner(path):
    path = os.path.abspath(path)
    dirname, filename = os.path.split(path)
    new_path = os.path.join(dirname, filename.split('.')[0])

    logger.INFO("Transforming '{}' -> '{}'".format(path, new_path))
    logger.INFO("Loading pickled file")
    old_runner = load_pickled_file(path)
    config = old_runner.config.copy()
    new_runner = CalibrationRunner(config.copy(), storage_path=new_path)

    for ii, step in enumerate(old_runner.experiments.keys()):
        storage_path = os.path.join(new_path, str(ii))
        experiment = old_runner.experiments[step]
        if experiment is None:
            continue

        # First copy traces, afterwards the pathes are changed
        logger.INFO("Copy traces for {}".format(step))
        for ii, measurement in enumerate(experiment.measurements):
            if measurement.traces is None:
                continue
            source = measurement.traces.fullpath
            target = os.path.join(storage_path, str(ii))
            logger.TRACE(" {} -> {}".format(source, target))
            shutil.copy2(source, target)

        logger.INFO("Create step {} at '{}'".format(step, storage_path))
        calib = CalibrationStep(config.copy(step),
                                storage_path,
                                old_runner.calibtic,
                                experiment=experiment)
        calib.result = old_runner.coeffs.get(step, None)
        calib.save()




def main():
    init_logger(pylogging.LogLevel.INFO, [
        ("main", pylogging.LogLevel.INFO),
        ("pycake", pylogging.LogLevel.DEBUG),
        ("pycake.calibrationrunner", pylogging.LogLevel.TRACE),
        ])

    parser = argparse.ArgumentParser(
            description='Convert old runner to newwer runner')
    add_logger_options(parser)
    parser.add_argument('runner', type=check_file)

    args = parser.parse_args()

    convert_old_runner(args.runner)

if __name__ == '__main__':
    main()
