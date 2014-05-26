import pyhalbe
import Coordinate
from pycake.helpers.units import linspace_voltage, DAC, Voltage, Current
import os
from itertools import product

np = neuron_parameter = pyhalbe.HICANN.neuron_parameter
sp = shared_parameter = pyhalbe.HICANN.shared_parameter


def pproduct(paramters, ranges):
    return [dict(zip(paramters, step)) for step in product(*ranges)]

from fastcalibration_parameters import parameters

folder = "/wang/users/koke/test_calib"

custom = {
        "filename_prefix":  "V_syntc_w_traces",

        "E_l_I_gl_fixed_range":      pproduct((np.E_l, np.I_gl), (
            linspace_voltage(550, 850, 6),
            [DAC(v) for v in (20, 40, 80, 120, 240, 480, 960)])),
        # For testing
        #     linspace_voltage(550, 850, 2),
        #     [DAC(v) for v in (20, 480)])),
        "V_syntcx_range": [{neuron_parameter.V_syntcx : v} for v in linspace_voltage(1240, 1600, 19)],

        "run_V_reset":         True,
        "run_E_synx":          True,
        "run_E_syni":          True,
        "run_V_t":             True,
        "run_E_l_I_gl_fixed":  True,
        "run_V_syntcx":        False,

        # Set whether you want to keep traces or delete them after analysis
        "save_traces":  False,
        "calibrate":    True,
        "measure":      True,

        # Where do you want to save the measurements (folder) and calibration
        # data (backend_c for calibtic, backend_r for redman)?
        # Folders will be created if they do not exist already
        "folder":       folder,
        "backend_c":    os.path.join(folder, "backends"),
        "backend_r":    os.path.join(folder, "backends"),

        # Wafer and HICANN coordinates
        "coord_wafer":  Coordinate.Wafer(),
        "coord_hicann": Coordinate.HICANNOnWafer(Coordinate.Enum(276)),

        "E_l_I_gl_fixed_parameters": {
            neuron_parameter.V_t:        Voltage(1200),
            neuron_parameter.E_syni:     Voltage(600, apply_calibration=True),
            neuron_parameter.E_synx:     Voltage(800, apply_calibration=True),
            shared_parameter.V_reset:    Voltage(500, apply_calibration=True),
        },

        "V_syntcx_parameters":  {
                                    neuron_parameter.E_l:     Voltage(700,
                                        apply_calibration=True),
                                    neuron_parameter.E_syni:  Voltage(600,
                                        apply_calibration=True),
                                    neuron_parameter.I_gl:    Current(1000,
                                        apply_calibration=True),
                                    neuron_parameter.E_synx:  Voltage(800,
                                        apply_calibration=True),
                                    shared_parameter.V_reset: Voltage(400,
                                        apply_calibration=True),
                                    shared_parameter.V_gmax0: Voltage(25),
                                    shared_parameter.V_gmax1: Voltage(25),
                                    shared_parameter.V_gmax2: Voltage(25),
                                    shared_parameter.V_gmax3: Voltage(25),
                                },

        # In which order should the parameters be calibrated?
        "parameter_order": [
            shared_parameter.V_reset.name,
            neuron_parameter.E_syni.name,
            neuron_parameter.E_synx.name,
            neuron_parameter.V_t.name,
            "E_l_I_gl_fixed",
            neuron_parameter.V_syntcx.name,
            ],
}
parameters.update(custom)

