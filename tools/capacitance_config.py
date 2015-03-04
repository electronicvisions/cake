import pyhalbe
from Coordinate import NeuronOnHICANN, FGBlockOnHICANN, Enum
from pycake.helpers.units import DAC, Voltage, Current
from pycake.helpers.units import linspace_voltage, linspace_current
import os
from itertools import product

import numpy as np

neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

folder = "/tmp"

def pproduct(paramters, ranges):
    return [dict(zip(paramters, step)) for step in product(*ranges)]

E_l_target = 900
amplitude = 200 # Sawtooth signal amplitude

V_reset = int(E_l_target - amplitude/2.)
V_t = int(E_l_target + amplitude/2.)

parameters = {
    # Which neurons and blocks do you want to calibrate?
    "filename_prefix": "",
    "neurons": [NeuronOnHICANN(Enum(i)) for i in range(512)],
    "blocks":  [FGBlockOnHICANN(Enum(i)) for i in range(4)],

    # Set the ranges within which you want to calibrate
    "capacitance_range": [{neuron_parameter.E_l: Voltage(E_l_target)}], # Set some realistic E_l value.

    # How many repetitions?
    # Each repetition will take about 1 minute per step!
    "repetitions":  1,

    # Set which calibrations you want to run
    "run_capacitance": True,

    # Smallcap?
    "smallcap": False,

    # Measurement runs twice by default: first to generate calibration data, and a second time to measure the success of calibration
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

    # Where do you want to save the measurements (folder) and calibration data (backend_c for calibtic, backend_r for redman)?
    # Folders will be created if they do not exist already
    "folder":       folder,
    "backend_c":    os.path.join(folder, "backends"),
    "backend_r":    os.path.join(folder, "backends"),

    # Wafer and HICANN coordinates
    "coord_wafer":  pyhalbe.Coordinate.Wafer(),
    "coord_hicann": pyhalbe.Coordinate.HICANNOnWafer(pyhalbe.Coordinate.Enum(276)),

    # Here you can set the fixed parameters for each calibration.
    # base_parameters are set for all calibrations
    "base_parameters":   {neuron_parameter.E_l: Voltage(E_l_target),
                          neuron_parameter.E_syni: Voltage(E_l_target),     # synapse
                          neuron_parameter.E_synx: Voltage(E_l_target),    # synapse
                          neuron_parameter.I_bexp: Current(2500),       # turn off exp by setting this to 2500 and see I_rexp and V_bexp
                          neuron_parameter.I_convi: Current(0),   # bias current for synaptic input
                          neuron_parameter.I_convx: Current(0),   # bias current for synaptic input
                          neuron_parameter.I_fire: Current(0),       # adaptation term b
                          neuron_parameter.I_gladapt: Current(0),    # adaptation term
                          neuron_parameter.I_gl: Current(0),      # leakage conductance
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
                          neuron_parameter.V_t: Voltage(V_t),       # recommended threshold, maximum is 1100

                          shared_parameter.V_reset:       Voltage(V_reset),   # * Neuron reset voltage
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

    'capacitance_parameters': {},

    # In which order should the parameters be calibrated?
    "parameter_order":      ['capacitance']}
