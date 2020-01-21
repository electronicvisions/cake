import pyhalbe
from pyhalco_common import Enum
from pyhalco_hicann_v2 import NeuronOnHICANN, FGBlockOnHICANN
from pycake.helpers.units import Volt, Ampere, Second
from pycake.helpers.units import linspace_voltage, linspace_current

neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

from fastcalibration_parameters import parameters

from fastcalibration_parameters import E_l_target
from fastcalibration_parameters import E_syni_target
from fastcalibration_parameters import E_synx_target
from fastcalibration_parameters import E_syn_distance

import numpy as np

# Time constants roughly correspond to ideal transformations of:
# I_pl = [10, 30, 50, 70, 90, 350, 2500] nA
tau_refs = [4.0e-6, 1.3e-6, 0.8e-6, 0.6e-6, 0.4e-6, 0.1e-6, 0.0]
# membrane time constants corresponding to ideal trafos of:
# I_gl = [100, 300, 900, 1500, 2500] nA
tau_mems = [1.654e-6, 1.126e-6, 7.171e-7, 6.851e-7, 4.543e-7]


eval_parameters = {
# Which neurons and blocks do you want to calibrate?
        "folder_prefix": "evaluation",
        "neurons": [NeuronOnHICANN(Enum(i)) for i in range(512)],
        "blocks":  [FGBlockOnHICANN(Enum(i)) for i in range(4)],

        "V_reset_range":  [{shared_parameter.V_reset : Volt(.5, apply_calibration=True)}],
        "E_syni_range":   [{neuron_parameter.E_syni :v } for v in linspace_voltage(E_syni_target, E_synx_target, 3, apply_calibration=True)],
        "E_synx_range":   [{neuron_parameter.E_synx :v } for v in linspace_voltage(E_syni_target, E_synx_target, 3, apply_calibration=True)],
        "E_l_range":      [{neuron_parameter.E_l : Volt(v, apply_calibration=True) } for v in [E_l_target-E_syn_distance/2, E_l_target, E_l_target+E_syn_distance/2]],
        "V_t_range":      [{neuron_parameter.V_t : v} for v in linspace_voltage(E_l_target, E_synx_target, 3, apply_calibration=True)],

        "I_gl_range":     [{neuron_parameter.I_gl : Second(tau_mem, apply_calibration=True)} for tau_mem in tau_mems],
        "V_syntcx_range": [{neuron_parameter.V_syntcx : Volt(1.44, apply_calibration=True)} ], # dummy
        "V_syntci_range": [{neuron_parameter.V_syntci : Volt(1.44, apply_calibration=True)} ], # dummy

        "I_pl_range":   [{neuron_parameter.I_pl : Second(t, apply_calibration=True)} for t in tau_refs],

        "Spikes_range" : [{neuron_parameter.V_t : Volt(v, apply_calibration=True)} for v in np.concatenate([[0.65, .66, .67], np.linspace(.68, .72, 11), [.73, .74, .75]])],

        # How many repetitions?
        # Each repetition will take about 1 minute per step!
        "repetitions":  1,

        # Set which calibrations you want to run
        "run_V_reset":  True,
        "run_E_synx":   True,
        "run_E_syni":   True,
        "run_E_l":      True,
        "run_V_t":      True,
        "run_I_gl":     False,
        "run_I_pl":     False,
        "run_Spikes" : False,

        # Measurement runs twice by default: first to generate calibration data, and a second time to measure the success of calibration
        # Here you can turn either of these runs on or off
        "calibrate":    False,
        "measure":      True,

        "Spikes_parameters": {
            neuron_parameter.E_l: Volt(E_l_target, apply_calibration=True),
            neuron_parameter.E_syni:     Volt(E_syni_target, apply_calibration=True),
            neuron_parameter.E_synx:     Volt(E_synx_target, apply_calibration=True),
            neuron_parameter.I_gl:       Ampere(1e-6, apply_calibration=True),
            neuron_parameter.I_pl:       Ampere(5e-9),
            neuron_parameter.V_syntcx: Volt(1.44, apply_calibration=True),  # dummy
            neuron_parameter.V_syntci: Volt(1.44, apply_calibration=True),  # dummy
            shared_parameter.V_reset:    Volt(0.2, apply_calibration=True),
        },

    # In which order should the parameters be evaluated?
    "parameter_order": [shared_parameter.V_reset.name,
                        neuron_parameter.E_syni.name,
                        neuron_parameter.E_synx.name,
                        neuron_parameter.E_l.name,
                        neuron_parameter.V_t.name,
                        neuron_parameter.I_gl.name,
                        neuron_parameter.I_pl.name,
                        "Spikes"
                       ],

}

parameters.update(eval_parameters)
