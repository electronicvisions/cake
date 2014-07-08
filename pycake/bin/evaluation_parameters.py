import pyhalbe
from Coordinate import NeuronOnHICANN, FGBlockOnHICANN, Enum
from pycake.helpers.units import Voltage, Current
from pycake.helpers.units import linspace_voltage, linspace_current

neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

folder = "/tmp"

from fastcalibration_parameters import parameters

from fastcalibration_parameters import E_l_target
from fastcalibration_parameters import E_syni_target
from fastcalibration_parameters import E_synx_target

eval_parameters = {
# Which neurons and blocks do you want to calibrate?
        "filename_prefix":  "",
        "neurons": [NeuronOnHICANN(Enum(i)) for i in range(512)],
        "blocks":  [FGBlockOnHICANN(Enum(i)) for i in range(4)],

        # For the evaluation of the calibration, only one point, except for V_t, is targeted
        "V_reset_range":  [{shared_parameter.V_reset : Voltage(500)}],
        "E_syni_range":   [{neuron_parameter.E_syni : Voltage(E_syni_target)}],
        "E_l_range":      [{neuron_parameter.E_l : Voltage(E_l_target)}],
        "V_t_range":      [{neuron_parameter.V_t : v} for v in linspace_voltage(E_l_target, E_synx_target, 3)],
        "E_synx_range":   [{neuron_parameter.E_synx : Voltage(E_synx_target)}],

        "I_gl_range":     [{neuron_parameter.I_gl : Current(0)}], # dummy
        "V_syntcx_range": [{neuron_parameter.V_syntcx : Voltage(1440)} ], # dummy
        "V_syntci_range": [{neuron_parameter.V_syntci : Voltage(1440)} ], # dummy

        # How many repetitions?
        # Each repetition will take about 1 minute per step!
        "repetitions":  1,

        # Set which calibrations you want to run
        "run_V_reset":  True,
        "run_E_synx":   True,
        "run_E_syni":   True,
        "run_E_l":      True,
        "run_V_t":      True,
        "run_I_gl":     True,
        "run_I_pl":     False,
        "run_V_syntcx": False,
        "run_V_syntci": False,
        "run_V_syntci_psp_max": False,
        "run_V_syntcx_psp_max": False,
        "run_E_l_I_gl_fixed":  False,

        # Measurement runs twice by default: first to generate calibration data, and a second time to measure the success of calibration
        # Here you can turn either of these runs on or off
        "calibrate":    False,
        "measure":      True,

}

parameters.update(eval_parameters)
