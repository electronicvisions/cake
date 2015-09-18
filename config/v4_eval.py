import numpy

from Coordinate import Enum
from pycake.helpers.units import DAC, Volt, Ampere
from pycake.helpers.units import linspace_voltage, linspace_current
from pyhalbe.HICANN import neuron_parameter
from pyhalbe.HICANN import shared_parameter


parameters = {
    "folder_prefix": "evaluation",

    # Run in measurement mode
    "calibrate":    False,
    "measure":      True,
    "clear_defects" : False,

    "V_reset_range":  [{
        shared_parameter.V_reset : Volt(v, apply_calibration=True),
        neuron_parameter.E_l : Volt(v + 0.4),
        neuron_parameter.V_t : Volt(v + 0.2)
    } for v in (0.6, 0.8)],

    "V_t_range":  [{
        shared_parameter.V_reset : Volt(v-200e-3),
        neuron_parameter.E_l : Volt(v + 200e-3),
        neuron_parameter.V_t : Volt(v, apply_calibration=True),
    } for v in (0.9, 1.0, 1.1)],

    "E_syni_range":   [{
        neuron_parameter.E_syni : Volt(v, apply_calibration=True),
        } for v in (0.7, 0.8)],

    "E_synx_range":   [{
        neuron_parameter.E_synx : Volt(v, apply_calibration=True),
        } for v in (1.0, 1.1)],

    "E_l_range": [{
        neuron_parameter.E_l : v,
        } for v in linspace_voltage(700e-3, 900e-3, 3)],

    "V_convoff_test_range": [
        {
            neuron_parameter.I_gl: Ampere(a * 1e-6),
        } for a in [0.0, 0.2, 0.3, 0.5, 1.0, 2.0]],

    "V_convoff_test_parameters": {
        neuron_parameter.E_l : Volt(0.8, apply_calibration=True),
        neuron_parameter.E_syni : Volt(0.6, apply_calibration=True),
        neuron_parameter.E_synx : Volt(1.3),
        neuron_parameter.V_convoffi : Volt(0.0, apply_calibration=True),
        neuron_parameter.V_convoffx : Volt(0.0, apply_calibration=True),
        neuron_parameter.V_t: Volt(1.12, apply_calibration=True),
        shared_parameter.V_reset:  Volt(0.3, apply_calibration=True),
    },

    "parameter_order": [
        'readout_shift',
        shared_parameter.V_reset.name,
        neuron_parameter.V_t.name,
        neuron_parameter.E_syni.name,
        neuron_parameter.E_synx.name,
        neuron_parameter.E_l,
        "V_convoff_test",
    ],
}

extends = ['v4_params.py']
