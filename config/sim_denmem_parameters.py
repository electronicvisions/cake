"""Simulation parameters for lib_denem_simulations"""

import os
import numpy
from itertools import product
from pyhalco_common import Enum
from pyhalco_hicann_v2 import FGBlockOnHICANN
from pyhalco_hicann_v2 import HICANNOnWafer
from pyhalco_hicann_v2 import NeuronOnHICANN
from pyhalco_hicann_v2 import Wafer
from pycake.helpers.units import Ampere
from pycake.helpers.units import DAC
from pycake.helpers.units import Volt
#from pycake.helpers.units import linspace_current
from pycake.helpers.units import linspace_voltage
from pyhalbe.HICANN import neuron_parameter
from pyhalbe.HICANN import shared_parameter
import pysthal


def pproduct(paramters, ranges):
    return [dict(zip(paramters, step)) for step in product(*ranges)]

# minimal floating gate current cell value [nA]
MIN_CUR_FG = 50e-9

MC = False
if MC:
    folder = "/wang/users/koke/cluster_home/calibration/sim_denmem_calib_mc2"
    MC_SEED = 24011985
    NEURONS = [NeuronOnHICANN(Enum(i)) for i in range(64)]
    BLOCKS = [FGBlockOnHICANN(Enum(i)) for i in range(2)]
else:
    folder = "/wang/users/koke/cluster_home/calibration/sim_denmem_calib"
    MC_SEED = None
    NEURONS = [NeuronOnHICANN(Enum(i)) for i in range(2)]
    BLOCKS = [FGBlockOnHICANN(Enum(i)) for i in range(2)]


# target resting potential
E_l_target = 900
E_synx_target = E_l_target + 300
E_syni_target = E_l_target - 300

parameters = {
    "folder_prefix": "sim_calibration",

    # host and port of your simulator server
    "sim_denmem": "vtitan:8123",  # host and port of your simulator

    # Cache folder, please use with care, changes on server side (e.g. switching
    # HICANN version, or pull) are not detected. Use None to deactivate,
    # folder must exist
    "sim_denmem_cache": "/fastnbig/home/koke/sim_denmem_cache/",

    # Use this seed for Monte Carlo simulations of neurons, None disables MC
    "sim_denmem_mc_seed" : MC_SEED,

    "coord_wafer": Wafer(0),  # required, determines MC seed
    "coord_hicann": HICANNOnWafer(Enum(0)),  # required, determines MC seed

    # HICANN version to use
    "hicann_version" : 4,

    "filename_prefix" : "base",

    "speedup": 'normal',                # options are fast, normal or slow
    "bigcap": True,

    "parameter_order": [
        shared_parameter.V_reset.name,
        neuron_parameter.V_t.name,
        neuron_parameter.E_syni.name,
        #neuron_parameter.E_synx.name,
        neuron_parameter.V_convoffx.name,
        neuron_parameter.V_convoffi.name,
        neuron_parameter.E_l.name,
        neuron_parameter.I_pl.name,
        "I_gl_charging",
    ],

    "folder":       folder,
    "backend":      os.path.join(folder, "backends"),

    "neurons": NEURONS,
    "blocks":  BLOCKS,
    "repetitions": 1,
    "save_traces": True,

    # Measurement runs twice by default: first to generate calibration data, and a second time to measure the success of calibration
    # Here you can turn either of these runs on or off
    "calibrate":    True,
    "measure":      False,

    # Overwrite old calibration data? This will not reset defect neurons!
    # Or even clear ALL calibration data before starting?
    "overwrite":    True,
    "clear":        True,

    # Set the ranges within which you want to calibrate
    "V_reset_range": [{shared_parameter.V_reset : Volt(v),
                       neuron_parameter.E_l : Volt(v + 400e-3),
                       neuron_parameter.V_t : Volt(v + 200e-3)}
                      for v in numpy.linspace(500e-3, 900e-3, 5)],
    "V_t_range": [{shared_parameter.V_reset : Volt(v-200e-3),
                   neuron_parameter.E_l : Volt(v + 200e-3),
                   neuron_parameter.V_t : Volt(v)}
                  for v in numpy.linspace(700e-3, 1100e-3, 5)],
    "E_syni_range": [{neuron_parameter.E_syni : v, neuron_parameter.E_l : v}
                     for v in linspace_voltage(400e-3, 800e-3, 5)],
    "E_synx_range": [{neuron_parameter.E_synx : v, neuron_parameter.E_l : v}
                     for v in linspace_voltage(1000e-3, 1400e-3, 5)],
    "V_convoffx_range": [{neuron_parameter.V_convoffx : v}
                         for v in linspace_voltage(400e-3, 1600e-3, 20)],
    # "V_convoffx_range": [{neuron_parameter.V_convoffx : v} for v in linspace_voltage(300e-3, 1500e-3, 3)],
    "V_convoffi_range": [{neuron_parameter.V_convoffi : v}
                         for v in linspace_voltage(400e-3, 1600e-3, 20)],
    "E_l_range": [{neuron_parameter.E_l : v} for v in linspace_voltage(550e-3, 850e-3, 6)],
    "I_pl_range": [{neuron_parameter.I_pl : Ampere(v*1e-9)} for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000, 1500, 2500]],

    "I_gl_charging_range": [
        {
            neuron_parameter.I_gl: Ampere(gl*1e-9),
            "speedup_gladapt": pysthal.NORMAL,
            "speedup_gl": pysthal.NORMAL,
        } for gl in [50, 75, 100, 500, 1000, 2000, 2500]
    ],

    # HICANN PLL
    "PLL": 125e6,

    # Here you can set the fixed parameters for each calibration.
    # base_parameters are set for all calibrations
    "base_parameters":  {
        neuron_parameter.E_l: Volt(900e-3),
        neuron_parameter.E_syni: Volt(700e-3),     # synapse
        neuron_parameter.E_synx: Volt(1100e-3),    # synapse
        neuron_parameter.I_bexp: Ampere(2500e-9),    # turn off exp by setting this to 2500 and see I_rexp and V_bexp
        neuron_parameter.V_convoffi: Volt(1000e-3), # Correction shift of internal synaptic integrator voltage
        neuron_parameter.V_convoffx: Volt(1000e-3), # Correction shift of internal synaptic integrator voltage
        neuron_parameter.I_convi: Ampere(2500e-9),   # bias current for synaptic input
        neuron_parameter.I_convx: Ampere(2500e-9),   # bias current for synaptic input
        neuron_parameter.I_fire: Ampere(0),       # adaptation term b
        neuron_parameter.I_gladapt: Ampere(0),    # adaptation term
        neuron_parameter.I_gl: Ampere(1000e-9),      # leakage conductance
        neuron_parameter.I_intbbi: Ampere(2000e-9),  # integrator bias in synapse
        neuron_parameter.I_intbbx: Ampere(2000e-9),  # integrator bias in synapse
        neuron_parameter.I_pl: Ampere(2000e-9),      # tau_refrac
        neuron_parameter.I_radapt: Ampere(2500e-9),  #
        neuron_parameter.I_rexp: Ampere(2500e-9),     # exp-term: must be 2500 if I_bexp is 2500
        neuron_parameter.I_spikeamp: Ampere(2000e-9),  #
        neuron_parameter.V_exp: Volt(1800e-3),      # exponential term
        neuron_parameter.V_syni: Volt(1000e-3),    # inhibitory synaptic reversal potential
        neuron_parameter.V_syntci: Volt(1420e-3),   # inhibitory synapse time constant
        neuron_parameter.V_syntcx: Volt(1420e-3),   # excitatory synapse time constant
        neuron_parameter.V_synx: Volt(1000e-3),    # excitatory synaptic reversal potential
        neuron_parameter.V_t: Volt(1000e-3),       # recommended threshold, maximum is 1100

        shared_parameter.V_reset:       Volt(500e-3),   # * Neuron reset voltage
        shared_parameter.int_op_bias:   DAC(1023),      # internal OP bias
        shared_parameter.V_dllres:      DAC(200),       # DLL reset voltage
        shared_parameter.V_bout:        Ampere(750e-9),   # * Global bias of neuron readout amps
        shared_parameter.V_bexp:        Ampere(2500e-9),  # * exp-term: Must be set to max if I_bexp is 2500
        shared_parameter.V_fac:         DAC(0),         # Short-term plasticity in facilitation mode
        shared_parameter.I_breset:      DAC(1023),      # Current used to pull down membrane after reset
        shared_parameter.V_dep:         DAC(0),         # Short-term plasticity in depression mode
        shared_parameter.I_bstim:       DAC(1023),      # Bias for neuron stimulation circuit
        shared_parameter.V_thigh:       DAC(0),         # STDP readout compare voltage
        shared_parameter.V_gmax3:       Volt(80e-3),    # max. synaptic weight
        shared_parameter.V_tlow:        DAC(0),         # STDP readout compare voltage (?)
        shared_parameter.V_gmax0:       Volt(80e-3),    # * max. synaptic weight
        shared_parameter.V_clra:        DAC(0),         # STDP CLR voltage (acausal)
        shared_parameter.V_clrc:        DAC(0),         # STDP CLR voltage (causa)l
        shared_parameter.V_gmax1:       Volt(80e-3),    # * max. synaptic weight
        shared_parameter.V_stdf:        DAC(0),         # STDF reset voltage
        shared_parameter.V_gmax2:       Volt(80e-3),    # * max. synaptic weight
        shared_parameter.V_m:           DAC(0),         # Start load voltage of causal STDP capacitor (grnd for acausal)
        shared_parameter.V_bstdf:       DAC(0),         # Bias for short-term plasticity
        shared_parameter.V_dtc:         DAC(0),         # Bias for DTC in short-term plasticity circuit
        shared_parameter.V_br:          DAC(0),         # Bias for STDP readout circuit
    },

    # parameters listed below will be written to the calibration with
    # the constant value set in base parameters
    "technical_parameters" : [neuron_parameter.I_convi,
                              neuron_parameter.I_convx,
                              neuron_parameter.I_intbbi,
                              neuron_parameter.I_intbbx,
                              neuron_parameter.V_syni,
                              neuron_parameter.V_synx,
                              neuron_parameter.I_spikeamp,
                              shared_parameter.I_breset,
                              shared_parameter.I_bstim,
                              shared_parameter.int_op_bias,
                              shared_parameter.V_bout,
                              shared_parameter.V_bexp,
                              shared_parameter.V_dllres],

    "V_reset_parameters":  {
        neuron_parameter.I_convi: Ampere(MIN_CUR_FG),
        neuron_parameter.I_convx: Ampere(MIN_CUR_FG),
        neuron_parameter.I_gl:   Ampere(1100e-9),
        neuron_parameter.I_pl:  Ampere(max(MIN_CUR_FG, 20e-9)),
        neuron_parameter.V_convoffi: Volt(1800e-3),
        neuron_parameter.V_convoffx: Volt(1800e-3),
    },

    "V_t_parameters": {
        neuron_parameter.I_convi: Ampere(MIN_CUR_FG),
        neuron_parameter.I_convx: Ampere(MIN_CUR_FG),
        neuron_parameter.I_gl:       Ampere(1500e-9),
        neuron_parameter.V_convoffi: Volt(1800e-3),
        neuron_parameter.V_convoffx: Volt(1800e-3),
    },

    "E_syni_parameters": {
        neuron_parameter.I_convi: Ampere(2500e-9),
        neuron_parameter.I_convx: Ampere(MIN_CUR_FG),
        neuron_parameter.I_gl: Ampere(MIN_CUR_FG),
        neuron_parameter.V_convoffi: Volt(300e-3),
        neuron_parameter.V_convoffx: Volt(1800e-3),
        neuron_parameter.V_t: Volt(1200e-3),
        shared_parameter.V_reset:  Volt(900e-3),
    },

    "E_synx_parameters": {
        # I_gl and I_convi MUST be set to min. realistic fg value
        neuron_parameter.I_convi: Ampere(MIN_CUR_FG),
        neuron_parameter.I_convx: Ampere(2500e-9),
        neuron_parameter.I_gl: Ampere(MIN_CUR_FG),
        neuron_parameter.V_convoffi: Volt(1800e-3),
        neuron_parameter.V_convoffx: Volt(300e-3),
        neuron_parameter.V_t: Volt(1200e-3),
        shared_parameter.V_reset:  Volt(900e-3),
    },

    "V_convoffx_parameters": {
        neuron_parameter.E_synx: Volt(1100e-3),
        neuron_parameter.E_l: Volt(900e-3),
        neuron_parameter.I_convi: Ampere(MIN_CUR_FG),
        neuron_parameter.I_convx: Ampere(2500e-9),
        neuron_parameter.I_gl: Ampere(1000e-9),
        neuron_parameter.V_convoffi: Volt(1800e-3),
        shared_parameter.V_gmax0: Ampere(2000e-9), # * max. synaptic weight
        neuron_parameter.V_t: Volt(1400e-3),
        shared_parameter.V_reset: Volt(900e-3), # Also initial membrane voltage in simulation
    },

    "V_convoffi_parameters": {
        neuron_parameter.E_syni: Volt(700e-3),
        neuron_parameter.E_l: Volt(900e-3),
        neuron_parameter.I_convi: Ampere(2500e-9),
        neuron_parameter.I_convx: Ampere(MIN_CUR_FG),
        neuron_parameter.I_gl: Ampere(1000e-9),
        neuron_parameter.V_convoffx: Volt(1800e-3),
        shared_parameter.V_gmax0: Ampere(2000e-9), # * max. synaptic weight
        neuron_parameter.V_t: Volt(1400e-3),
        shared_parameter.V_reset: Volt(900e-3), # Also initial membrane voltage in simulation
    },

    "E_l_parameters": {
        neuron_parameter.V_t: Volt(1200e-3),
        neuron_parameter.I_gl: Ampere(1000e-9),
        shared_parameter.V_reset: Volt(500e-3, apply_calibration=True),

        # calibrated V_convoff is a constant, value here should be irrelevant
        neuron_parameter.V_convoffi: Volt(1800e-3, apply_calibration=True),
        neuron_parameter.V_convoffx: Volt(1800e-3, apply_calibration=True),
    },


    "I_pl_parameters": {
        neuron_parameter.E_l: Volt(1200e-3),
        neuron_parameter.V_t: Volt(800e-3),
        shared_parameter.V_reset: Volt(500e-3, apply_calibration=True),
    },

    "I_gl_charging_parameters": {
        neuron_parameter.E_l:        Volt(800e-3, apply_calibration=True),
        neuron_parameter.V_t:        Volt(770e-3, apply_calibration=True),
        shared_parameter.V_reset:    Volt(700e-3, apply_calibration=True),

        neuron_parameter.I_gladapt: Ampere(MIN_CUR_FG),
        neuron_parameter.I_fire: Ampere(2500e-9),       # adaptation term b

        # calibrated V_convoff is a constant, value here should be irrelevant
        neuron_parameter.V_convoffi: Volt(1800e-3, apply_calibration=True),
        neuron_parameter.V_convoffx: Volt(1800e-3, apply_calibration=True),
    }

}
