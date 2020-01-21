"""
Calibration parameters for the HICANN v4.
"""

import os
import numpy

from pyhalco_common import Enum
from pycake.helpers.units import DAC, Volt, Ampere
from pycake.helpers.units import linspace_voltage, linspace_current
from pyhalbe.HICANN import neuron_parameter
from pyhalbe.HICANN import shared_parameter

extends = ['base_parameters.py']
folder = "/tmp"

parameters = {
    # Where do you want to save the measurements (folder) and calibration data 
    # (backend_c for calibtic, backend_r for redman)?
    # Folders will be created if they do not exist already
    "folder":       folder,
    "backend":      os.path.join(folder, "backends"),

    # Which neurons and blocks do you want to calibrate?
    "folder_prefix": "calibration",
    "calibrate":    True,
    # "save_traces":  True,
    "clear_defects" : True,

    "parameter_order": [
        'readout_shift',
        shared_parameter.V_reset.name,
        neuron_parameter.V_t.name,
        neuron_parameter.E_syni.name,
        neuron_parameter.E_synx.name, # Caveat: https://brainscales-r.kip.uni-heidelberg.de/issues/1929
        neuron_parameter.I_pl.name,
        neuron_parameter.E_l.name,
        neuron_parameter.V_convoffx.name,
        neuron_parameter.V_convoffi.name,
        #neuron_parameter.I_gl.name,
        "I_gl_PSP",
        neuron_parameter.V_syntcx.name,
        neuron_parameter.V_syntci.name,
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

    # E_l
    "E_l_range": [
        {
            neuron_parameter.E_l : v,
        } for v in linspace_voltage(700e-3, 900e-3, 3)
    ],

    "E_l_parameters": {
        neuron_parameter.I_convi: Ampere(0.0),
        neuron_parameter.I_convx: Ampere(0.0),
        neuron_parameter.V_t: Volt(1.2, apply_calibration=True),
        shared_parameter.V_reset:  Volt(0.9, apply_calibration=True),
    },

    # E_syni
    "E_syni_range": [
        {
            neuron_parameter.E_syni : v,
        } for v in linspace_voltage(400e-3, 800e-3, 3) # 5
    ],

    "E_syni_parameters": {
        neuron_parameter.E_l : Volt(0.8, apply_calibration=True),
        neuron_parameter.I_convx: Ampere(0.0),
        neuron_parameter.I_gl: Ampere(0.0e-6),
        neuron_parameter.V_convoffi: Volt(0.1),
        neuron_parameter.V_syntci: Volt(1.8), # Fast
        neuron_parameter.V_syntcx: Volt(1.8), # Fast
        neuron_parameter.V_t: Volt(1.2, apply_calibration=True),
        shared_parameter.V_reset:  Volt(0.9, apply_calibration=True),
    },

    # E_synx
    "E_synx_range": [
        {
            neuron_parameter.E_synx : v,
        } for v in linspace_voltage(0.8, 1.0, 3)
    ],

    "E_synx_parameters": {
        neuron_parameter.E_l : Volt(0.8, apply_calibration=True),
        neuron_parameter.I_convi: Ampere(0.0),
        neuron_parameter.I_gl: Ampere(0.0),
        neuron_parameter.V_convoffx: Volt(0.1),
        neuron_parameter.V_syntci: Volt(1.8), # Fast
        neuron_parameter.V_syntcx: Volt(1.8), # Fast
        neuron_parameter.V_t: Volt(1.2, apply_calibration=True),
        shared_parameter.V_reset:  Volt(0.9, apply_calibration=True),
    },

    # V_convoffi
    "V_convoffi_range": [
        {
            neuron_parameter.V_convoffi: Volt(v),
        } for v in numpy.linspace(0.4, 1.4, 25)
    ],

    "V_convoffi_parameters": {
        neuron_parameter.E_l : Volt(0.8, apply_calibration=True),
        neuron_parameter.E_syni : Volt(0.4, apply_calibration=True),
        # E_synx should not have a large influence because we set I_convx to 0
        neuron_parameter.E_synx : Volt(1.2),
        neuron_parameter.I_convx: Ampere(0.0),
        neuron_parameter.I_gl: Ampere(0.2e-6),
        neuron_parameter.V_convoffx: Volt(1.8),
        neuron_parameter.V_t: Volt(1.2, apply_calibration=True),
        # Low to make spiking traces easier to detect
        shared_parameter.V_reset: Volt(0.3, apply_calibration=True),
    },

    # V_convoffx 
    "V_convoffx_range": [
        {
            neuron_parameter.V_convoffx: Volt(v),
        } for v in numpy.linspace(0.4, 1.4, 25)
    ],

    "V_convoffx_parameters": {
        neuron_parameter.E_l : Volt(0.8, apply_calibration=True),
        neuron_parameter.E_synx : Volt(1.2),
        # E_syni should not have a large influence because we set I_convi to 0
        neuron_parameter.E_syni : Volt(0.4, apply_calibration=True),
        neuron_parameter.I_convi: Ampere(0),
        neuron_parameter.I_gl: Ampere(0.2e-6),
        neuron_parameter.V_convoffi: Volt(1.8),
        neuron_parameter.V_t: Volt(1.2, apply_calibration=True),
        # Low to make spiking traces easier to detect
        shared_parameter.V_reset: Volt(0.3, apply_calibration=True),
    },

    # V_syntci
    "V_syntci_range": [
        {
            neuron_parameter.V_syntci: Volt(v),
        } for v in numpy.concatenate(
            (numpy.logspace(numpy.log10(0.4), numpy.log10(0.9), 7),
             numpy.linspace(0.9, 1.8, 4)[1:]))
    ],

    "V_syntci_parameters": {
        neuron_parameter.E_l : Volt(0.8, apply_calibration=True),
        neuron_parameter.E_syni : Volt(0.6, apply_calibration=True),
        neuron_parameter.E_synx : Volt(1.3),
        neuron_parameter.I_gl: Ampere(0.3e-6),
        neuron_parameter.V_convoffi : Volt(1.8, apply_calibration=True),
        neuron_parameter.V_convoffx : Volt(1.8, apply_calibration=True),
        neuron_parameter.V_t: Volt(1.2, apply_calibration=True),
        shared_parameter.V_gmax0: Volt(0.05),
        shared_parameter.V_reset: Volt(0.3, apply_calibration=True),
        "gmax_div": 30,
    },

    # V_syntcx
    "V_syntcx_range": [
        {
            neuron_parameter.V_syntcx: Volt(v),
        } for v in numpy.concatenate(
            (numpy.logspace(numpy.log10(0.4), numpy.log10(0.9), 7),
             numpy.linspace(0.9, 1.8, 4)[1:]))
    ],

    "V_syntcx_parameters": {
        neuron_parameter.E_l : Volt(0.8, apply_calibration=True),
        neuron_parameter.E_syni : Volt(0.6, apply_calibration=True),
        neuron_parameter.E_synx : Volt(1.3),
        neuron_parameter.I_gl: Ampere(0.3e-6),
        neuron_parameter.V_convoffi : Volt(1.8, apply_calibration=True),
        neuron_parameter.V_convoffx : Volt(1.8, apply_calibration=True),
        neuron_parameter.V_t: Volt(1.2, apply_calibration=True),
        shared_parameter.V_gmax0: Volt(0.05),
        shared_parameter.V_reset: Volt(0.3, apply_calibration=True),
        "gmax_div": 30,
    },

    # values between 10 and 100 are suitable
    # [10,15,20,30,40,60,80] is a good set for calib
    "I_pl_range":   [
        {
            neuron_parameter.I_pl : DAC(d)
        } for d in [10, 15, 20, 30, 40, 60, 80]
    ],

    "I_pl_parameters": {
        neuron_parameter.E_l: Volt(1.2),
        neuron_parameter.V_t:        Volt(0.8),
        neuron_parameter.E_syni:     Volt(0.8, apply_calibration=True),
        neuron_parameter.E_synx:     Volt(1.2),
        shared_parameter.V_reset:    Volt(0.5, apply_calibration=True),
    },

    "I_pl_repetitions": 4,

    # I_gl
    "I_gl_range": [
        {
            neuron_parameter.I_gl : Ampere(v),
        } for v in numpy.linspace(100e-9, 800e-9, 8)
    ],

    "I_gl_parameters": {
        neuron_parameter.E_l: Volt(0.8, apply_calibration=True),
        neuron_parameter.V_t: Volt(1.2),
        neuron_parameter.E_syni: Volt(0.6, apply_calibration=True),
        neuron_parameter.E_synx: Volt(1.3),
        shared_parameter.V_reset:    Volt(.2, apply_calibration=True),
        neuron_parameter.V_convoffi : Volt(1.8, apply_calibration=True),
        neuron_parameter.V_convoffx : Volt(1.8, apply_calibration=True),
        neuron_parameter.I_convi: Ampere(0.0),
        neuron_parameter.I_convx: Ampere(0.0),
    },

    "I_gl_PSP_range": [
        {
            neuron_parameter.I_gl : Ampere(v),
        } for v in numpy.linspace(100e-9, 800e-9, 8)
    ],


    "I_gl_PSP_parameters": {
        neuron_parameter.E_l : Volt(0.8, apply_calibration=True),
        neuron_parameter.E_syni : Volt(0.6, apply_calibration=True),
        neuron_parameter.E_synx : Volt(1.3),
        neuron_parameter.V_syntcx: Volt(1.6),
        neuron_parameter.V_convoffi : Volt(0.9, apply_calibration=True),
        neuron_parameter.V_convoffx : Volt(0.9, apply_calibration=True),
        neuron_parameter.V_t: Volt(1.2, apply_calibration=True),
        shared_parameter.V_gmax0: Volt(1),
        shared_parameter.V_reset: Volt(0.3, apply_calibration=True),
        "gmax_div": 30,
        "bigcap" : True,
        "speedup_I_gl" : "normal"
    },
}
