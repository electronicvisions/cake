"""
Import all cake submodules here.

Useful for test coverage report.
Also ensures that dependencies are installed.

Generate a list running
$ tree pycake -P '*.py'
"""

import unittest
import pycake

# base modules
import pycake.analyzer
import pycake.calibrationrunner
import pycake.calibrator
import pycake.config
import pycake.experimentbuilder
import pycake.experiment
import pycake.measure
import pycake.reader

# calibration submodule
import pycake.calibration
import pycake.calibration.capacitance
import pycake.calibration.exponential
import pycake.calibration.E_l_I_gl_fixed
import pycake.calibration.I_gl_charging

# helpers submodule
import pycake.helpers
import pycake.helpers.calibtic
import pycake.helpers.misc
import pycake.helpers.peakdetect
import pycake.helpers.psp_fit
import pycake.helpers.psp_shapes
import pycake.helpers.redman
import pycake.helpers.SignalToNoise
import pycake.helpers.sthal
import pycake.helpers.StorageProcess
import pycake.helpers.TraceAverager
import pycake.helpers.TracesOnDiskDict
import pycake.helpers.units
import pycake.helpers.WorkerPool

# logic submodule
import pycake.logic
import pycake.logic.psps
import pycake.logic.spikes
import pycake.logic.utils

# visualization submodule
import pycake.visualization
import pycake.visualization.plotting


class TestImport(unittest.TestCase):
    def test_nothing(self):
        """Do nothing, this just exists to get rid of a
        'no tests executed' message.
        Maybe all imports should move in here, because this file
        tests if everything can be imported without errors.
        """
        pass
