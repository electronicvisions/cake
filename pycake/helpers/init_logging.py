"""
common logger initialization routine for cake tools
"""

# Log multiprocessing output to stderr
import argparse
import logging
import multiprocessing

import pylogging

from pysthal.command_line_util import add_logger_options
from pysthal.command_line_util import init_logger

class PyLoggingHandler(logging.Handler):
    LOGLEVELS = {
        logging.CRITICAL: pylogging.Logger.fatal,
        logging.ERROR: pylogging.Logger.error,
        logging.WARNING: pylogging.Logger.warn,
        logging.INFO: pylogging.Logger.info,
        logging.DEBUG: pylogging.Logger.debug,
    }

    def emit(self, record):
        if record.processName is None:
            logger = pylogging.get(record.name)
        else:
            logger = pylogging.get(record.name + '.' + record.processName)
        f = self.LOGLEVELS.get(record.levelno, pylogging.Logger.fatal)
        f(logger, record.getMessage())

def init_cake_logging(overwrite_defaults=[]):
    init_logger(pylogging.LogLevel.WARN, [
        ("Default", pylogging.LogLevel.INFO),
        ("halbe.fgwriter", pylogging.LogLevel.ERROR),
        ("progress", pylogging.LogLevel.DEBUG),
        ("progress.sthal", pylogging.LogLevel.INFO),
        ("pycake.analyzer", pylogging.LogLevel.TRACE),
        ("pycake.calibrationrunner", pylogging.LogLevel.DEBUG),
        ("pycake.calibrationunit", pylogging.LogLevel.DEBUG),
        ("pycake.calibtic", pylogging.LogLevel.DEBUG),
        ("pycake.config", pylogging.LogLevel.INFO),
        ("pycake.experiment", pylogging.LogLevel.DEBUG),
        ("pycake.experimentbuilder", pylogging.LogLevel.DEBUG),
        ("pycake.helper.simsthal", pylogging.LogLevel.INFO),
        ("pycake.helper.sthal", pylogging.LogLevel.INFO),
        ("pycake.helper.workerpool", pylogging.LogLevel.INFO),
        ("pycake.measurement", pylogging.LogLevel.DEBUG),
        ("pycake.redman", pylogging.LogLevel.DEBUG),
        ("sthal", pylogging.LogLevel.INFO),
        ("sthal.AnalogRecorder", pylogging.LogLevel.WARN),
        ("sthal.HICANNConfigurator.Time", pylogging.LogLevel.INFO),
        (multiprocessing.util.LOGGER_NAME, pylogging.LogLevel.DEBUG),
        ] + overwrite_defaults)

    logger = multiprocessing.get_logger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(PyLoggingHandler())

    parser = argparse.ArgumentParser(add_help=False)
    add_logger_options(parser)
    args, _ = parser.parse_known_args()

    if args.logfile is not None:
        progress = pylogging.get('progress')
        pylogging.append_to_cout(progress)
        progress.warn("Write logs to {}".format(args.logfile))
