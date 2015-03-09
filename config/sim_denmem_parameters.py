"""Simulation parameters for lib_denem_simulations"""

import os
import numpy
from itertools import product
from Coordinate import Enum
from Coordinate import FGBlockOnHICANN
from Coordinate import HICANNOnWafer
from Coordinate import NeuronOnHICANN
from Coordinate import Wafer
from pycake.helpers.units import Current
from pycake.helpers.units import DAC
from pycake.helpers.units import Voltage
#from pycake.helpers.units import linspace_current
from pycake.helpers.units import linspace_voltage
from pyhalbe.HICANN import neuron_parameter
from pyhalbe.HICANN import shared_parameter
import pysthal


def pproduct(paramters, ranges):
    return [dict(zip(paramters, step)) for step in product(*ranges)]

# minimal floating gate current cell value [nA]
MIN_CUR_FG = 50

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

    "parameter_order": [
        shared_parameter.V_reset.name,
        neuron_parameter.V_t.name,
        neuron_parameter.E_syni.name,
        #neuron_parameter.E_synx.name,
        neuron_parameter.V_convoffx.name,
        neuron_parameter.V_convoffi.name,
        neuron_parameter.I_pl.name,
        "I_gl_charging",
    ],

    "folder":       folder,
    "backend_c":    os.path.join(folder, "backends"),

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
    "V_reset_range": [{shared_parameter.V_reset : Voltage(v),
                       neuron_parameter.E_l : Voltage(v + 400),
                       neuron_parameter.V_t : Voltage(v + 200)}
                      for v in numpy.linspace(500, 900, 5)],
    "V_t_range": [{shared_parameter.V_reset : Voltage(v-200),
                   neuron_parameter.E_l : Voltage(v + 200),
                   neuron_parameter.V_t : Voltage(v)}
                  for v in numpy.linspace(700, 1100, 5)],
    "E_syni_range": [{neuron_parameter.E_syni : v, neuron_parameter.E_l : v}
                     for v in linspace_voltage(400, 800, 5)],
    "E_synx_range": [{neuron_parameter.E_synx : v, neuron_parameter.E_l : v}
                     for v in linspace_voltage(1000, 1400, 5)],
    "V_convoffx_range": [{neuron_parameter.V_convoffx : v}
                         for v in linspace_voltage(400, 1600, 20)],
    # "V_convoffx_range": [{neuron_parameter.V_convoffx : v} for v in linspace_voltage(300, 1500, 3)],
    "V_convoffi_range": [{neuron_parameter.V_convoffi : v}
                         for v in linspace_voltage(400, 1600, 20)],

    "I_pl_range": [{neuron_parameter.I_pl : Current(v)} for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000, 1500, 2500]],

    "I_gl_charging_range": [
        {
            neuron_parameter.I_gl: Current(gl),
            "speedup_gladapt": pysthal.NORMAL,
            "speedup_gl": pysthal.NORMAL,
        } for gl in [50, 75, 100, 500, 1000, 2000, 2500]
    ],

    # HICANN PLL
    "PLL": 125e6,

    # Here you can set the fixed parameters for each calibration.
    # base_parameters are set for all calibrations
    "base_parameters":  {
        neuron_parameter.E_l: Voltage(900),
        neuron_parameter.E_syni: Voltage(700),     # synapse
        neuron_parameter.E_synx: Voltage(1100),    # synapse
        neuron_parameter.I_bexp: Current(2500),    # turn off exp by setting this to 2500 and see I_rexp and V_bexp
        neuron_parameter.V_convoffi: Voltage(1000), # Correction shift of internal synaptic integrator voltage
        neuron_parameter.V_convoffx: Voltage(1000), # Correction shift of internal synaptic integrator voltage
        neuron_parameter.I_convi: Current(2500),   # bias current for synaptic input
        neuron_parameter.I_convx: Current(2500),   # bias current for synaptic input
        neuron_parameter.I_fire: Current(0),       # adaptation term b
        neuron_parameter.I_gladapt: Current(0),    # adaptation term
        neuron_parameter.I_gl: Current(1000),      # leakage conductance
        neuron_parameter.I_intbbi: Current(2000),  # integrator bias in synapse
        neuron_parameter.I_intbbx: Current(2000),  # integrator bias in synapse
        neuron_parameter.I_pl: Current(2000),      # tau_refrac
        neuron_parameter.I_radapt: Current(2500),  #
        neuron_parameter.I_rexp: Current(2500),     # exp-term: must be 2500 if I_bexp is 2500
        neuron_parameter.I_spikeamp: Current(2000),  #
        neuron_parameter.V_exp: Voltage(1800),      # exponential term
        neuron_parameter.V_syni: Voltage(1000),    # inhibitory synaptic reversal potential
        neuron_parameter.V_syntci: Voltage(1420),   # inhibitory synapse time constant
        neuron_parameter.V_syntcx: Voltage(1420),   # excitatory synapse time constant
        neuron_parameter.V_synx: Voltage(1000),    # excitatory synaptic reversal potential
        neuron_parameter.V_t: Voltage(1000),       # recommended threshold, maximum is 1100

        shared_parameter.V_reset:       Voltage(500),   # * Neuron reset voltage
        shared_parameter.int_op_bias:   DAC(1023),      # internal OP bias
        shared_parameter.V_dllres:      DAC(200),       # DLL reset voltage
        shared_parameter.V_bout:        Current(750),   # * Global bias of neuron readout amps
        shared_parameter.V_bexp:        Current(2500),  # * exp-term: Must be set to max if I_bexp is 2500
        shared_parameter.V_fac:         DAC(0),         # Short-term plasticity in facilitation mode
        shared_parameter.I_breset:      DAC(1023),      # Current used to pull down membrane after reset
        shared_parameter.V_dep:         DAC(0),         # Short-term plasticity in depression mode
        shared_parameter.I_bstim:       DAC(1023),      # Bias for neuron stimulation circuit
        shared_parameter.V_thigh:       DAC(0),         # STDP readout compare voltage
        shared_parameter.V_gmax3:       Voltage(80),    # max. synaptic weight
        shared_parameter.V_tlow:        DAC(0),         # STDP readout compare voltage (?)
        shared_parameter.V_gmax0:       Voltage(80),    # * max. synaptic weight
        shared_parameter.V_clra:        DAC(0),         # STDP CLR voltage (acausal)
        shared_parameter.V_clrc:        DAC(0),         # STDP CLR voltage (causa)l
        shared_parameter.V_gmax1:       Voltage(80),    # * max. synaptic weight
        shared_parameter.V_stdf:        DAC(0),         # STDF reset voltage
        shared_parameter.V_gmax2:       Voltage(80),    # * max. synaptic weight
        shared_parameter.V_m:           DAC(0),         # Start load voltage of causal STDP capacitor (grnd for acausal)
        shared_parameter.V_bstdf:       DAC(0),         # Bias for short-term plasticity
        shared_parameter.V_dtc:         DAC(0),         # Bias for DTC in short-term plasticity circuit
        shared_parameter.V_br:          DAC(0),         # Bias for STDP readout circuit
    },

    "V_reset_parameters":  {
        neuron_parameter.I_convi: Current(MIN_CUR_FG),
        neuron_parameter.I_convx: Current(MIN_CUR_FG),
        neuron_parameter.I_gl:   Current(1100),
        neuron_parameter.I_pl:  Current(max(MIN_CUR_FG, 20)),
        neuron_parameter.V_convoffi: Voltage(1800),
        neuron_parameter.V_convoffx: Voltage(1800),
    },

    "V_t_parameters": {
        neuron_parameter.I_convi: Current(MIN_CUR_FG),
        neuron_parameter.I_convx: Current(MIN_CUR_FG),
        neuron_parameter.I_gl:       Current(1500),
        neuron_parameter.V_convoffi: Voltage(1800),
        neuron_parameter.V_convoffx: Voltage(1800),
    },

    "E_syni_parameters": {
        neuron_parameter.I_convi: Current(2500),
        neuron_parameter.I_convx: Current(MIN_CUR_FG),
        neuron_parameter.I_gl: Current(MIN_CUR_FG),
        neuron_parameter.V_convoffi: Voltage(300),
        neuron_parameter.V_convoffx: Voltage(1800),
        neuron_parameter.V_t: Voltage(1200),
        shared_parameter.V_reset:  Voltage(900),
    },

    "E_synx_parameters": {
        # I_gl and I_convi MUST be set to min. realistic fg value
        neuron_parameter.I_convi: Current(MIN_CUR_FG),
        neuron_parameter.I_convx: Current(2500),
        neuron_parameter.I_gl: Current(MIN_CUR_FG),
        neuron_parameter.V_convoffi: Voltage(1800),
        neuron_parameter.V_convoffx: Voltage(300),
        neuron_parameter.V_t: Voltage(1200),
        shared_parameter.V_reset:  Voltage(900),
    },

    "V_convoffx_parameters": {
        neuron_parameter.E_synx: Voltage(1100),
        neuron_parameter.E_l: Voltage(900),
        neuron_parameter.I_convi: Current(MIN_CUR_FG),
        neuron_parameter.I_convx: Current(2500),
        neuron_parameter.I_gl: Current(1000),
        neuron_parameter.V_convoffi: Voltage(1800),
        shared_parameter.V_gmax0: Current(2000), # * max. synaptic weight
        neuron_parameter.V_t: Voltage(1400),
        shared_parameter.V_reset: Voltage(900), # Also initial membrane voltage in simulation
    },

    "V_convoffi_parameters": {
        neuron_parameter.E_syni: Voltage(700),
        neuron_parameter.E_l: Voltage(900),
        neuron_parameter.I_convi: Current(2500),
        neuron_parameter.I_convx: Current(MIN_CUR_FG),
        neuron_parameter.I_gl: Current(1000),
        neuron_parameter.V_convoffx: Voltage(1800),
        shared_parameter.V_gmax0: Current(2000), # * max. synaptic weight
        neuron_parameter.V_t: Voltage(1400),
        shared_parameter.V_reset: Voltage(900), # Also initial membrane voltage in simulation
    },


    "I_pl_parameters": {
        neuron_parameter.E_l: Voltage(1200),
        neuron_parameter.V_t: Voltage(800),
        neuron_parameter.E_syni: Voltage(E_syni_target, apply_calibration=True),
        neuron_parameter.E_synx: Voltage(E_synx_target, apply_calibration=True),
        shared_parameter.V_reset: Voltage(500, apply_calibration=True),
    },

    "I_gl_charging_parameters": {
        neuron_parameter.E_l:        Voltage(800, apply_calibration=True),
        neuron_parameter.V_t:        Voltage(770, apply_calibration=True),
        shared_parameter.V_reset:    Voltage(700, apply_calibration=True),

        neuron_parameter.I_gladapt: Current(MIN_CUR_FG),
        neuron_parameter.I_fire: Current(2500),       # adaptation term b

        # calibrated V_convoff is a constant, value here should be irrelevant
        neuron_parameter.V_convoffi: Voltage(1800, apply_calibration=True),
        neuron_parameter.V_convoffx: Voltage(1800, apply_calibration=True),
    }

}
