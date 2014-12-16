"""
Import all cake submodules here.

Useful for test coverage report.
Also ensures that dependencies are installed.
"""

import pycake

# base modules
import pycake.analyzer
import pycake.calibrator
import pycake.experimentbuilder
import pycake.reader
import pycake.calibrationrunner
import pycake.config
import pycake.experiment
import pycake.measure

# calibration submodule
import pycake.calibration
import pycake.calibration.E_l_I_gl_fixed
import pycake.calibration.vsyntc

# helpers submodule
import pycake.helpers
import pycake.helpers.peakdetect
import pycake.helpers.SignalToNoise
import pycake.helpers.TracesOnDiskDict
import pycake.helpers.calibtic
import pycake.helpers.psp_fit
import pycake.helpers.sthal
import pycake.helpers.trafos
import pycake.helpers.psp_shapes
import pycake.helpers.StorageProcess
import pycake.helpers.units
import pycake.helpers.misc
import pycake.helpers.redman
import pycake.helpers.TraceAverager
import pycake.helpers.WorkerPool

# logic submodule
import pycake.logic
import pycake.logic.helpers
import pycake.logic.psps
import pycake.logic.spikes
import pycake.logic.taum_fit
