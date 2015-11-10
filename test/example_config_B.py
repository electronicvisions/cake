"""Example configuration used by test_config.py"""

import Coordinate
from pyhalbe.HICANN import shared_parameter
from pycake.helpers.units import DAC

parameters = {
    "wafer_cfg" : "foo.xml",
    "parameter_order" : ["foo", "bar", "baz"],
    "sim_denmem" : "vtitan:8123",
    "PLL" : 125e6
}
