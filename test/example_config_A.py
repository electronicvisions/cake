"""Example configuration used by test_config.py"""

from pyhalco_common import iter_all, Enum
import pyhalco_hicann_v2 as Coordinate
from pyhalbe.HICANN import shared_parameter
from pycake.helpers.units import DAC

parameters = {
    "speedup": 'normal',
    "bigcap": True,
    "blocks": [block for block in iter_all(Coordinate.FGBlockOnHICANN)],
    "neurons": [neuron for neuron in iter_all(Coordinate.NeuronOnHICANN)],
    "random-name_foo": "foo-value",
    "random-name_range": "steps-value",
    "parameter_order": [],
    "base_parameters": {shared_parameter.int_op_bias: DAC(1023),
                        shared_parameter.V_dllres: DAC(200)},
    "technical_parameters": [shared_parameter.int_op_bias,
                            shared_parameter.V_dllres],
    "folder": "folder-value",
    "coord_wafer": Coordinate.Wafer(17),
    "coord_hicann": Coordinate.HICANNOnWafer(Enum(222)),
    "backend": "backend-value",
    "clear": "clear-value",
    "calibrate": "calibrate-value",
    "measure": "measure-value",
}
