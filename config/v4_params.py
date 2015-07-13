"""
Calibration parameters for the HICANN v4.
"""

import os
import numpy

from Coordinate import Enum
from pycake.helpers.units import DAC, Volt, Ampere
from pycake.helpers.units import linspace_voltage, linspace_current
from pyhalbe.HICANN import neuron_parameter
from pyhalbe.HICANN import shared_parameter

folder = "/wang/users/koke/cluster_home/calibration/v4_vsetup1"

parameters = {
    # Where do you want to save the measurements (folder) and calibration data 
    # (backend_c for calibtic, backend_r for redman)?
    # Folders will be created if they do not exist already
    "folder":       folder,
    "backend_c":    os.path.join(folder, "backends"),
    "backend_r":    os.path.join(folder, "backends"),

    # Which neurons and blocks do you want to calibrate?
    "folder_prefix": "calibration",
    "calibrate":    True,

    "parameter_order": [
        'readout_shift',
        shared_parameter.V_reset.name,
        neuron_parameter.V_t.name,
        neuron_parameter.E_syni.name,
        neuron_parameter.E_synx.name,

        # neuron_parameter.V_convoffx.name,
        # neuron_parameter.V_convoffi.name,
    ],

    # readout shift: Set some realistic E_l value.
    "readout_shift_range": [
        {neuron_parameter.E_l: Volt(0.9)},
    ],

    "readout_shift_parameters": {
        neuron_parameter.V_t: Volt(1.4),
        neuron_parameter.I_convx: Ampere(0),
        neuron_parameter.I_convi: Ampere(0),
    },

    # V_reset
    "V_reset_range": [
        {
            shared_parameter.V_reset : Volt(v),
            neuron_parameter.E_l : Volt(v + 0.4),
            neuron_parameter.V_t : Volt(v + 0.2)
        } for v in numpy.linspace(0.5, 0.8, 4)],

    "V_reset_parameters":  {
        neuron_parameter.I_convi: Ampere(0.0),
        neuron_parameter.I_convx: Ampere(0.0),
        neuron_parameter.I_gl:   Ampere(1100e-9),
        neuron_parameter.I_pl:  Ampere(20e-9),
    },

    # V_t
    "V_t_range": [
        {
            shared_parameter.V_reset : Volt(v-200e-3),
            neuron_parameter.E_l : Volt(v + 200e-3),
            neuron_parameter.V_t : Volt(v),
        } for v in numpy.linspace(700e-3, 1100e-3, 5)
    ],

    "V_t_parameters": {
        neuron_parameter.I_convi: Ampere(0.0),
        neuron_parameter.I_convx: Ampere(0.0),
        neuron_parameter.I_gl: Ampere(1500e-9),
    },

    # E_syni
    "E_syni_range": [
        {
            neuron_parameter.E_syni : v,
            neuron_parameter.E_l : v
        } for v in linspace_voltage(400e-3, 800e-3, 5)
    ],

    "E_syni_parameters": {
        neuron_parameter.I_convx: Ampere(0.0),
        neuron_parameter.I_gl: Ampere(0.0),
        neuron_parameter.V_convoffi: Volt(0.3),
        neuron_parameter.V_t: Volt(1.2),
        shared_parameter.V_reset:  Volt(0.9),
    },

    # E_synx
    "E_synx_range": [
        {
            neuron_parameter.E_synx : v,
            neuron_parameter.E_l : v
        } for v in linspace_voltage(0.8, 1.0, 3)
    ],

    "E_synx_parameters": {
        neuron_parameter.I_convx: Ampere(0.0),
        neuron_parameter.I_gl: Ampere(0.0),
        neuron_parameter.V_convoffi: Volt(0.3),
        neuron_parameter.V_t: Volt(1.2),
        shared_parameter.V_reset:  Volt(0.9),
    },

    # values between 10 and 100 can be used, 2500 is for zero refractory time
    "I_pl_range":   [
        {
            neuron_parameter.I_pl : Ampere(1e-9*I)
        } for I in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 2500]
    ],
}

extends = ['base_parameters.py']
