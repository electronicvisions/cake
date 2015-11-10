
#/usr/bin/env python
# -*- coding: utf-8 -*-

from pyhalbe.HICANN import neuron_parameter
from pycake.analyzer import Analyzer
from pycake.measure import ADCMeasurement
from pycake.experimentbuilder import BaseExperimentBuilder
from pycake.helpers.peakdetect import peakdet

class I_rexp_Analyzer(Analyzer):
    pass


class V_exp_Analyzer(Analyzer):
    pass


class I_rexp_Experimentbuilder(BaseExperimentBuilder):
    """Calibration method for \Delta_T by Marco Schwartz
    as described in section 4.3.3 of his thesis.

    This method analyzes the current I_exp from the exponential term
    to the membrane, which is not available in hardware measurements.
    """
    def get_analyzer(self):
        return I_rexp_Analyzer()
