"""Example configuration used by test_config.py"""

import Coordinate

parameters = {
    "blocks": "blocks-value",
    "random-name_foo": "foo-value",
    "random-name_range": "steps-value",
    "parameter_order": [],
    "neurons": "neurons-value",
    "folder": "folder-value",
    "coord_wafer": Coordinate.Wafer(17),
    "coord_hicann": Coordinate.HICANNOnWafer(Coordinate.Enum(222)),
    "backend_c": "backend-value",
    "clear": "clear-value",
    "calibrate": "calibrate-value",
    "measure": "measure-value",
}
