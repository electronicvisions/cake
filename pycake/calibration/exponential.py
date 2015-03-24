
#/usr/bin/env python
# -*- coding: utf-8 -*-

from pyhalbe.HICANN import neuron_parameter
from pycake.analyzer import Analyzer
from pycake.measure import ADCMeasurement
from pycake.experimentbuilder import BaseExperimentBuilder
from pycake.helpers.peakdetect import peakdet
from pycake.helpers.units import Voltage


class I_rexp_Analyzer(Analyzer):
    pass


class V_exp_Analyzer(Analyzer):
    pass


class I_rexp_Experimentbuilder(BaseExperimentBuilder):
    def get_analyzer(self):
        return I_rexp_Analyzer()
