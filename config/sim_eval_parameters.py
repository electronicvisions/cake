"""Evaluate calibration in MonteCarlo transistor level simulation"""

import numpy

from pycake.helpers.units import Volt, Ampere, Second
from pycake.helpers.misc import nested_update

from pyhalbe.HICANN import neuron_parameter
from pyhalbe.HICANN import shared_parameter

# original calibration config
from sim_denmem_parameters import parameters

# Time constants roughly correspond to ideal transformations of:
# I_pl = [10, 30, 50, 70, 90, 350, 2500] nA
tau_refs = [4.0e-6, 1.3e-6, 0.8e-6, 0.6e-6, 0.4e-6, 0.1e-6, 0.0]

eval_parameters = {
    "folder_prefix": "sim_evaluation",

    # For the evaluation of the calibration, sometimes only one point is
    # targeted, usually 3
    "V_reset_range": [{shared_parameter.V_reset : Volt(500e-3),
                       neuron_parameter.E_l : Volt(900e-3),
                       neuron_parameter.V_t : Volt(700e-3)}],
    "E_l_range": [{neuron_parameter.E_l: Volt(v)} for v in [650e-3, 700e-3, 750e-3]],
    "V_t_range": [{shared_parameter.V_reset : Volt(v-200e-3),
                   neuron_parameter.E_l: Volt(v + 200e-3),
                   neuron_parameter.V_t: Volt(v)} for v in numpy.linspace(700e-3, 1100e-3, 3)],
    "I_gl_charging_range": [{neuron_parameter.I_gl : Ampere(c)} for c in [500e-9, 700e-9]],
    "I_pl_range": [{neuron_parameter.I_pl : Second(t)} for t in tau_refs],

    # How many repetitions?
    # Each repetition will take about 1 minute per step!
    "repetitions": 1,

    # Set which evaluations you want to run
    "run_V_reset": True,
    "run_V_t": True,
    "run_E_syni": True,
    "run_E_synx": False,  # 2015-03-11: not calibrated
    "run_V_convoffx": False,  # constant value
    "run_V_convoffi": False,  # constant value
    "run_E_l": True,
    "run_I_pl": True,
    "run_I_gl_charging": True,
    "run_V_syntcx": False,  # no calibration
    "run_V_syntci": False,  # no calibration

    # Do not calibrate, just measure for evaluation
    "calibrate": False,
    "measure": True,
}

# overwrite ranges by regular update
parameters.update(eval_parameters)


special_parameters = {
    "base_parameters": {
        neuron_parameter.V_convoffi: Volt(1800e-3, apply_calibration=True),
        neuron_parameter.V_convoffx: Volt(1800e-3, apply_calibration=True),
    },
    "V_reset_parameters": {
        neuron_parameter.V_convoffi: Volt(1800e-3, apply_calibration=True),
        neuron_parameter.V_convoffx: Volt(1800e-3, apply_calibration=True),
    }
}

# modify base parameter without losing other base parameters
nested_update(parameters, special_parameters)
