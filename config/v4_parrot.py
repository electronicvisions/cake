"""
Parameters to find suitable neurons for L1 tests
"""

import os
import numpy

from pyhalco_common import Enum
from pycake.helpers.units import DAC, Volt, Ampere
from pycake.helpers.units import linspace_voltage, linspace_current
from pyhalbe.HICANN import neuron_parameter
from pyhalbe.HICANN import shared_parameter
folder = "/wang/users/koke/cluster_home/calibration/v4_vsetup1_tmp4"

parameters = {
    # Where do you want to save the measurements (folder) and calibration data 
    # (backend_c for calibtic, backend_r for redman)?
    # Folders will be created if they do not exist already
    "folder":       folder,
    "backend":      os.path.join(folder, "backends"),

   # Which neurons and blocks do you want to calibrate?
    "folder_prefix": "parrot",
    "calibrate":    True,

    "parameter_order": [
        'Parrot',
    ],

    "Parrot_range": [
        {
            neuron_parameter.I_gl: Ampere(2.0e-6),
        } for ii in range(10)
    ],

    "Parrot_parameters": {
        neuron_parameter.E_l : Volt(0.8, apply_calibration=True),
        neuron_parameter.E_syni : Volt(0.6, apply_calibration=True),
        neuron_parameter.E_synx : Volt(1.3),
        neuron_parameter.I_pl: DAC(10),
        neuron_parameter.V_convoffi : Volt(1.8, apply_calibration=True),
        neuron_parameter.V_convoffx : Volt(1.8, apply_calibration=True),
        neuron_parameter.V_syntcx: Volt(0.55),
        neuron_parameter.V_t: Volt(1.2, apply_calibration=True),
        shared_parameter.V_gmax0: Volt(0.2),
        shared_parameter.V_reset: Volt(0.3, apply_calibration=True),
        "gmax_div": 2,
    },
}

extends = ['base_parameters.py']
