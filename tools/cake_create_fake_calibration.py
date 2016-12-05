#!/usr/bin/env python

import pycalibtic
import Coordinate as C
from pycake.helpers import calibtic

c = calibtic.Calibtic(".", C.Wafer(), C.HICANNOnWafer(C.Enum(280)))
backend = c.get_backend()
c.hc.setDefaults()
backend.store(c.get_calibtic_name(), pycalibtic.MetaData(), c.hc)
