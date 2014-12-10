import pyhalbe
from Coordinate import NeuronOnHICANN, FGBlockOnHICANN, Enum
from pycake.helpers.units import Voltage, Current
from pycake.helpers.units import linspace_voltage, linspace_current

neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

from fastcalibration_parameters import parameters

from fastcalibration_parameters import E_l_target
from fastcalibration_parameters import E_syni_target
from fastcalibration_parameters import E_synx_target
from fastcalibration_parameters import E_syn_distance

import numpy as np

eval_parameters = {
# Which neurons and blocks do you want to calibrate?
        "filename_prefix":  "",
        "neurons": [NeuronOnHICANN(Enum(i)) for i in range(512)],
        "blocks":  [FGBlockOnHICANN(Enum(i)) for i in range(4)],

        # For the evaluation of the calibration, only one point, except for V_t, is targeted
        "V_reset_range":  [{shared_parameter.V_reset : Voltage(500)}],
        "E_syni_range":   [{neuron_parameter.E_syni :v } for v in linspace_voltage(E_syni_target, E_synx_target, 3)],
        "E_synx_range":   [{neuron_parameter.E_synx :v } for v in linspace_voltage(E_syni_target, E_synx_target, 3)],
        "E_l_range":      [{neuron_parameter.E_l : Voltage(v) } for v in [E_l_target-E_syn_distance/2, E_l_target, E_l_target+E_syn_distance/2]],
        "V_t_range":      [{neuron_parameter.V_t : v} for v in linspace_voltage(E_l_target, E_synx_target, 3)],

        "I_gl_range":     [{neuron_parameter.I_gl : Current(0)}], # dummy
        "V_syntcx_range": [{neuron_parameter.V_syntcx : Voltage(1440)} ], # dummy
        "V_syntci_range": [{neuron_parameter.V_syntci : Voltage(1440)} ], # dummy

        "I_pl_range":   [{neuron_parameter.I_pl : Current(1.0/v)} for v in [10, 30, 50, 70, 90, 2500]],

        "spikes_range" : [{neuron_parameter.V_t : Voltage(500)}],

        # How many repetitions?
        # Each repetition will take about 1 minute per step!
        "repetitions":  1,

        # Set which calibrations you want to run
        "run_V_reset":  False,
        "run_E_synx":   False,
        "run_E_syni":   False,
        "run_E_l":      False,
        "run_V_t":      True,
        "run_I_gl":     False,
        "run_I_pl":     False,
        "run_V_syntcx": False,
        "run_V_syntci": False,
        "run_V_syntci_psp_max": False,
        "run_V_syntcx_psp_max": False,
        "run_E_l_I_gl_fixed":  False,
        "run_spikes" : False,

        # Measurement runs twice by default: first to generate calibration data, and a second time to measure the success of calibration
        # Here you can turn either of these runs on or off
        "calibrate":    False,
        "measure":      True,

        "spikes_parameters": {
            neuron_parameter.E_l: Voltage(E_l_target, apply_calibration=True),
            neuron_parameter.E_syni:     Voltage(E_syni_target, apply_calibration=True),
            neuron_parameter.E_synx:     Voltage(E_synx_target, apply_calibration=True),
            neuron_parameter.I_gl:       Current(1000, apply_calibration=True),
            neuron_parameter.V_syntcx: Voltage(1440, apply_calibration=True),  # dummy
            neuron_parameter.V_syntci: Voltage(1440, apply_calibration=True),  # dummy
            shared_parameter.V_reset:    Voltage(500, apply_calibration=True),
        },


}

parameters.update(eval_parameters)
