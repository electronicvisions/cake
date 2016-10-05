"""
Base parameters for all calibrations.

The default is to run in calibration mode and to keep existing calibration
values.
"""

from pycake.helpers.units import Ampere
from pycake.helpers.units import DAC
from pycake.helpers.units import Volt
from pyhalbe.HICANN import neuron_parameter
from pyhalbe.HICANN import shared_parameter

from Coordinate import Enum
from Coordinate import FGBlockOnHICANN
from Coordinate import NeuronOnHICANN

parameters = {
    # Measurement runs twice by default: first to generate calibration data,
    # and a second time to measure the success of calibration
    # Here you can turn either of these runs on or off
    "calibrate":    True,
    "measure":      False,

    # Overwrite old calibration data? This will not reset defect neurons!
    # Or even clear ALL calibration data before starting?
    "overwrite":    True,
    "clear":        False,
    "clear_defects" : False,

    # Set whether you want to keep traces or delete them after analysis
    "save_traces":  False,
    "V_reset_save_traces":  True,
    "V_t_save_traces":  True,

    # Selecet specific neurons only
    "neurons": [NeuronOnHICANN(Enum(i)) for i in range(512)],
    "blocks":  [FGBlockOnHICANN(Enum(i)) for i in range(4)],

    # options are fast, normal or slow
    "speedup": 'normal',
    "bigcap": True,

    # Default floating gate biases, MAXIMUM POWER!
    "fg_bias": 0,
    "fg_biasn": 0,

    # Here you can set the fixed parameters for each calibration.
    # base_parameters are set for all calibrations
    "base_parameters":   {
        neuron_parameter.E_l: Volt(0.8),
        neuron_parameter.E_syni: Volt(0.6),     # synapse
        neuron_parameter.E_synx: Volt(1.3),    # synapse
        neuron_parameter.I_bexp: Ampere(2500e-9),       # turn off exp by setting this to 2500 and see I_rexp and V_bexp
        neuron_parameter.I_convi: Ampere(625e-9),   # bias current for synaptic input
        neuron_parameter.I_convx: Ampere(625e-9),   # bias current for synaptic input
        neuron_parameter.I_fire: Ampere(0),       # adaptation term b
        neuron_parameter.I_gladapt: Ampere(0),    # adaptation term
        neuron_parameter.I_gl: Ampere(1000e-9),      # leakage conductance
        neuron_parameter.I_intbbi: Ampere(2000e-9),  # integrator bias in synapse
        neuron_parameter.I_intbbx: Ampere(2000e-9),  # integrator bias in synapse
        neuron_parameter.I_pl: Ampere(2000e-9),      # tau_refrac
        neuron_parameter.I_radapt: Ampere(2500e-9),  #
        neuron_parameter.I_rexp: Ampere(2500e-9),     # exp-term: must be 2500 if I_bexp is 2500
        neuron_parameter.I_spikeamp: Ampere(2000e-9),  #
        neuron_parameter.V_convoffi: Volt(1.8), # Correction shift of internal synaptic integrator voltage
        neuron_parameter.V_convoffx: Volt(1.8), # Correction shift of internal synaptic integrator voltage
        neuron_parameter.V_exp: Volt(1.8),      # exponential term
        neuron_parameter.V_syni: Volt(0.9),    # inhibitory synaptic reversal potential
        neuron_parameter.V_syntci: Volt(1.42),   # inhibitory synapse time constant
        neuron_parameter.V_syntcx: Volt(1.42),   # excitatory synapse time constant
        neuron_parameter.V_synx: Volt(0.9),    # excitatory synaptic reversal potential
        neuron_parameter.V_t: Volt(1.),       # recommended threshold, maximum is 1.1

        shared_parameter.V_reset:       Volt(0.5),   # * Neuron reset voltage
        shared_parameter.int_op_bias:   DAC(1023),      # internal OP bias
        shared_parameter.V_dllres:      DAC(200),       # DLL reset voltage
        shared_parameter.V_ccas:        DAC(800),       # layer 1 amplifier bias
        shared_parameter.V_bout:        Ampere(750e-9),   # * Global bias of neuron readout amps
        shared_parameter.V_bexp:        Ampere(2500e-9),  # * exp-term: Must be set to max if I_bexp is 2500
        shared_parameter.V_fac:         DAC(0),         # Short-term plasticity in facilitation mode
        shared_parameter.I_breset:      DAC(1023),      # Current used to pull down membrane after reset
        shared_parameter.V_dep:         DAC(0),         # Short-term plasticity in depression mode
        shared_parameter.I_bstim:       DAC(1023),      # Bias for neuron stimulation circuit
        shared_parameter.V_thigh:       DAC(0),         # STDP readout compare voltage
        shared_parameter.V_gmax3:       Volt(.080),    # max. synaptic weight
        shared_parameter.V_tlow:        DAC(0),         # STDP readout compare voltage (?)
        shared_parameter.V_gmax0:       Volt(.080),    # * max. synaptic weight
        shared_parameter.V_clra:        DAC(0),         # STDP CLR voltage (acausal)
        shared_parameter.V_clrc:        DAC(0),         # STDP CLR voltage (causa)l
        shared_parameter.V_gmax1:       Volt(.080),    # * max. synaptic weight
        shared_parameter.V_stdf:        DAC(0),         # STDF reset voltage
        shared_parameter.V_gmax2:       Volt(.080),    # * max. synaptic weight
        shared_parameter.V_m:           DAC(0),         # Start load voltage of causal STDP capacitor (grnd for acausal)
        shared_parameter.V_bstdf:       DAC(0),         # Bias for short-term plasticity
        shared_parameter.V_dtc:         DAC(0),         # Bias for DTC in short-term plasticity circuit
        shared_parameter.V_br:          DAC(0),         # Bias for STDP readout circuit
    },

    # parameters listed below will be written to the calibration with
    # the constant value set in base parameters
    "technical_parameters" : [
         neuron_parameter.I_convi,
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
         shared_parameter.V_dllres,
         shared_parameter.V_ccas,
    ],
}
