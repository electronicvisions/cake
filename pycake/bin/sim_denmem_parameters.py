"""Simulation parameters for lib_denem_simulations"""

import os
from Coordinate import Enum
from Coordinate import FGBlockOnHICANN
from Coordinate import HICANNOnWafer
from Coordinate import NeuronOnHICANN
from Coordinate import Wafer
from pycake.helpers.units import Current
from pycake.helpers.units import DAC
from pycake.helpers.units import Voltage
from pycake.helpers.units import linspace_voltage, linspace_current
from pyhalbe.HICANN import neuron_parameter
from pyhalbe.HICANN import shared_parameter

folder = "/tmp"

parameters = {
    # host and port of your simulator server
    "sim_denmem": "vtitan:8123",  # host and port of your simulator

    # Cache folder, please use with care, changes on server side (e.g. switching
    # HICANN version, or pull) are not detected. Use None to deactivate, 
    # folder must exits
    "sim_denmem_cache": None,

    # Use this seed for monte-carlos simulations of neurons, None disables MC
    "sim_denmem_mc_seed" : None,

    "coord_wafer": Wafer(0),  # required, determines monte-carlo seed
    "coord_hicann": HICANNOnWafer(Enum(0)),  # required, determines monte-carlo seed

    "parameter_order": [
        shared_parameter.V_reset.name,
        neuron_parameter.V_t.name,
        neuron_parameter.E_syni.name,
        neuron_parameter.E_synx.name,
        ],

    # empty tempdirs, TODO: delete afterwards
    "folder":       folder,
    "backend_c":    os.path.join(folder, "backends"),
    "backend_r":    os.path.join(folder, "backends"),

    "neurons": [NeuronOnHICANN(Enum(i)) for i in range(2)],
    "blocks":  [FGBlockOnHICANN(Enum(i)) for i in range(1)],
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
    "V_reset_range":  [{shared_parameter.V_reset : v} for v in linspace_voltage(600, 800, 5)],
    "E_syni_range":   [{neuron_parameter.E_syni : v} for v in linspace_voltage(550, 850, 5)],
    "E_synx_range":   [{neuron_parameter.E_synx : v} for v in linspace_voltage(650, 950, 5)],
    "V_t_range":      [{neuron_parameter.V_t : v} for v in linspace_voltage(600, 900, 4)],

    # HICANN PLL
    "PLL": 125e6,

    # Here you can set the fixed parameters for each calibration.
    # base_parameters are set for all calibrations
    "base_parameters":  {
        neuron_parameter.E_l: Voltage(900),
        neuron_parameter.E_syni: Voltage(700),     # synapse
        neuron_parameter.E_synx: Voltage(1100),    # synapse
        neuron_parameter.I_bexp: Current(2500),    # turn off exp by setting this to 2500 and see I_rexp and V_bexp
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
        neuron_parameter.E_l:    Voltage(1300),
        neuron_parameter.I_convi: Current(0),
        neuron_parameter.I_convx: Current(0),
        neuron_parameter.I_gl:   Current(1100),
        neuron_parameter.I_pl:  Current(20),
        neuron_parameter.V_t:    Voltage(900),
    },

    "V_t_parameters": {
        neuron_parameter.E_l:        Voltage(1200),
        neuron_parameter.I_convi: Current(0),
        neuron_parameter.I_convx: Current(0),
        neuron_parameter.I_gl:       Current(1500),
        shared_parameter.V_reset:    Voltage(400),
    },

    "E_syni_parameters": {
        neuron_parameter.I_convi: Current(2500),
        neuron_parameter.I_convx: Current(0),
        neuron_parameter.I_gl: Current(0),  # I_gl and I_convx MUST be set to 0
        neuron_parameter.V_t: Voltage(1200),
        shared_parameter.V_reset:  Voltage(200),
    },

    "E_synx_parameters": {
        neuron_parameter.I_convi: Current(0),
        neuron_parameter.I_convx: Current(2500),
        neuron_parameter.I_gl: Current(0),  # I_gl and I_convi MUST be set to 0
        neuron_parameter.V_t: Voltage(1200),
        shared_parameter.V_reset:  Voltage(200),
        #neuron_parameter.V_syntcx: Voltage(1440, apply_calibration=True), # dummy
        #neuron_parameter.V_syntci: Voltage(1440, apply_calibration=True), # dummy
    },
}
