#!/usr/bin/env python

import pycalibtic
from pyhalco_common import Enum
import pyhalco_hicann_v2 as C
from pycake.helpers import calibtic

c = calibtic.Calibtic(".", C.Wafer(), C.HICANNOnWafer(Enum(280)))
backend = c.get_backend()
c.hc.setDefaults()
backend.store(c.get_calibtic_name(), pycalibtic.MetaData(), c.hc)
