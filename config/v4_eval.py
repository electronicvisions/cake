import numpy

from pyhalco_common import Enum
from pycake.helpers.units import DAC, Volt, Ampere, Second
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
        neuron_parameter.V_convoffi : Volt(0.0, apply_calibration=True),
        neuron_parameter.V_convoffx : Volt(0.0, apply_calibration=True),
        } for v in linspace_voltage(700e-3, 900e-3, 3, apply_calibration=True)],

    "V_convoff_test_uncalibrated_range": [
        {
            neuron_parameter.I_gl: Ampere(a * 1e-6),
        } for a in [0.0, 0.2, 0.3, 0.5, 1.0, 2.0]],

    # with uncalibrated V_convoffi/x at typical value
    "V_convoff_test_uncalibrated_parameters": {
        neuron_parameter.E_l : Volt(0.8, apply_calibration=True),
        neuron_parameter.E_syni : Volt(0.6, apply_calibration=True),
        neuron_parameter.E_synx : Volt(1.3),
        neuron_parameter.V_convoffi : Volt(0.9, apply_calibration=False),
        neuron_parameter.V_convoffx : Volt(0.9, apply_calibration=False),
        neuron_parameter.V_t: Volt(1.12, apply_calibration=True),
        shared_parameter.V_reset:  Volt(0.3, apply_calibration=True),
    },

    "V_convoff_test_calibrated_range": [
        {
            neuron_parameter.I_gl: Ampere(a * 1e-6),
        } for a in [0.0, 0.2, 0.3, 0.5, 1.0, 2.0]],

    "V_convoff_test_calibrated_parameters": {
        neuron_parameter.E_l : Volt(0.8, apply_calibration=True),
        neuron_parameter.E_syni : Volt(0.6, apply_calibration=True),
        neuron_parameter.E_synx : Volt(1.3),
        neuron_parameter.V_convoffi : Volt(0.0, apply_calibration=True),
        neuron_parameter.V_convoffx : Volt(0.0, apply_calibration=True),
        neuron_parameter.V_t: Volt(1.12, apply_calibration=True),
        shared_parameter.V_reset:  Volt(0.3, apply_calibration=True),
    },

    "V_syntcx_range": [
        {neuron_parameter.V_syntcx: Second(t, apply_calibration=True),
        }
        for t in [6e-7, 5e-7, 4e-7, 3e-7]
    ],

    "V_syntci_range": [
        {neuron_parameter.V_syntci: Second(t, apply_calibration=True),
        }
        for t in [6e-7, 5e-7, 4e-7, 3e-7]
    ],

    "I_pl_range":   [
        {neuron_parameter.I_pl : Second(t, apply_calibration=True)}
        for t in [0.1e-6, 0.2e-6, 0.3e-6, 0.4e-6, 0.5e-6]
    ],

    "I_pl_repetitions": 1,

    "I_gl_range": [{neuron_parameter.I_gl : Second(t, apply_calibration=True)} for t in [1e-6]],

    "I_gl_PSP_parameters": {
        neuron_parameter.E_l : Volt(0.8, apply_calibration=True),
        neuron_parameter.E_syni : Volt(0.6, apply_calibration=True),
        neuron_parameter.E_synx : Volt(1.3),
        neuron_parameter.V_convoffi : Volt(0.9, apply_calibration=True),
        neuron_parameter.V_convoffx : Volt(0.9, apply_calibration=True),
        neuron_parameter.V_t: Volt(1.2, apply_calibration=True),
        shared_parameter.V_gmax0: Volt(1),
        shared_parameter.V_reset: Volt(0.3, apply_calibration=True),
        "gmax_div": 30,
    },

    "I_gl_PSP_range": [

        {neuron_parameter.V_syntcx: Second(1e-7, apply_calibration=True),
         neuron_parameter.I_gl : Second(75e-7, apply_calibration=True),
         "bigcap" : True, "speedup_I_gl" : "slow"},

        {neuron_parameter.V_syntcx: Second(1e-7, apply_calibration=True),
         neuron_parameter.I_gl : Second(50e-7, apply_calibration=True),
         "bigcap" : True, "speedup_I_gl" : "slow"},

        {neuron_parameter.V_syntcx: Second(1e-7, apply_calibration=True),
         neuron_parameter.I_gl : Second(10e-7, apply_calibration=True),
         "bigcap" : True, "speedup_I_gl" : "normal"},

        {neuron_parameter.V_syntcx: Second(1e-7, apply_calibration=True),
         neuron_parameter.I_gl : Second(5e-7, apply_calibration=True),
         "bigcap" : False, "speedup_I_gl" : "normal"},

        {neuron_parameter.V_syntcx: Second(1e-7, apply_calibration=True),
         neuron_parameter.I_gl : Second(10e-7, apply_calibration=True),
         "bigcap" : True, "speedup_I_gl" : "fast"},

    ],

    "Spikes_parameters": {
        neuron_parameter.E_l: Volt(0.8, apply_calibration=True),
        neuron_parameter.E_syni:     Volt(0.6, apply_calibration=True),
        neuron_parameter.E_synx:     Volt(1.3),
        neuron_parameter.I_gl:       Second(10e-7, apply_calibration=True),
        neuron_parameter.I_pl:       Second(1e-7, apply_calibration=True),
        neuron_parameter.V_convoffx: Volt(1.8, apply_calibration=True),
        neuron_parameter.V_convoffi: Volt(1.8, apply_calibration=True),
        shared_parameter.V_reset:    Volt(0.2, apply_calibration=True),
        "bigcap" : True,
        "speedup_I_gl" : "normal"
    },

    "Spikes_range" : [{neuron_parameter.V_t : Volt(v, apply_calibration=True)}
                      for v in numpy.concatenate([[.70, .75, .76, .77],
                                                  numpy.linspace(.78, .82, 11),
                                                  [.83, .84, .85, 0.9]])],

    "parameter_order": [
        'readout_shift', # dummy eval. cf. https://brainscales-r.kip.uni-heidelberg.de/projects/cake/work_packages/3276/
        shared_parameter.V_reset.name, # V_reset must come before V_t for plotting script to work
        neuron_parameter.V_t.name,
        neuron_parameter.E_syni.name,
        neuron_parameter.E_synx.name,
        neuron_parameter.E_l.name,
        "V_convoff_test_uncalibrated",
        "V_convoff_test_calibrated",
        neuron_parameter.V_syntcx.name,
        neuron_parameter.V_syntci.name,
        neuron_parameter.I_pl.name,
        "I_gl_PSP",
        "Spikes"
    ],
}

extends = ['v4_params.py']
