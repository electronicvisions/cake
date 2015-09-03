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
        } for v in linspace_voltage(700e-3, 900e-3, 3)
    ],

    "parameter_order": [
        'readout_shift',
        shared_parameter.V_reset.name,
        neuron_parameter.V_t.name,
        neuron_parameter.E_syni.name,
        neuron_parameter.E_synx.name,

        # neuron_parameter.V_convoffx.name,
        # neuron_parameter.V_convoffi.name,
    ],
}

extends = ['v4_params.py']
