import pyhalbe
from pyhalco_common import Enum
from pyhalco_hicann_v2 import Wafer, HICANNOnWafer, NeuronOnHICANN, FGBlockOnHICANN
from pycake.helpers.units import DAC, Volt, Ampere
from pycake.helpers.units import linspace_voltage, linspace_current
import os
from itertools import product

import numpy as np

neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

folder = "/tmp"


def pproduct(paramters, ranges):
    return [dict(zip(paramters, step)) for step in product(*ranges)]

E_l_target = 0.7  # FIXME: propagate to E_l_I_gl_fixed calibrator
E_syn_distance = 0.1

E_syni_target = E_l_target - E_syn_distance
E_synx_target = E_l_target + E_syn_distance

parameters = {
    # Which neurons and blocks do you want to calibrate?
    "folder_prefix": "calibration",
    "speedup": 'normal',                # options are fast, normal or slow
    "bigcap": True,
    "neurons": [NeuronOnHICANN(Enum(i)) for i in range(512)],
    "blocks":  [FGBlockOnHICANN(Enum(i)) for i in range(4)],

    "hicann_version" : 2,
    "fg_bias": 8,
    "fg_biasn": 5,

    # Set the ranges within which you want to calibrate
    "readout_shift_range": [{neuron_parameter.E_l: Volt(0.9)}], # Set some realistic E_l value.
    "V_reset_range":  [{shared_parameter.V_reset : v} for v in linspace_voltage(0.6, 0.8, 5)],
    "E_syni_range":   [{neuron_parameter.E_syni : v} for v in linspace_voltage(0.55, 0.85, 5)],
    "E_synx_range":   [{neuron_parameter.E_synx : v} for v in linspace_voltage(0.65, 0.95, 5)],
    "E_l_range":      [{neuron_parameter.E_l : v} for v in linspace_voltage(0.55, 0.85, 6)],
    "V_t_range":      [{neuron_parameter.V_t : v} for v in linspace_voltage(0.6, 0.9, 4)],
    "E_l_I_gl_fixed_range": pproduct((neuron_parameter.E_l, neuron_parameter.I_gl),
                                     (linspace_voltage(0.55, 0.85, 2), [DAC(v) for v in (40, 80, 120, 240, 480, 960)])),
    "I_gl_range":     [{neuron_parameter.I_gl : v} for v in linspace_current(100e-9, 800e-9, 8)],
    "V_syntcx_range": [{neuron_parameter.V_syntcx : v} for v in linspace_voltage(1.2, 1.66, 20)],
    "V_syntcx_psp_max_range": [{neuron_parameter.V_syntcx : v} for v in linspace_voltage(1.35, 1.7, 10)],
    "V_syntci_psp_max_range": [{neuron_parameter.V_syntci : v} for v in linspace_voltage(1.35, 1.7, 10)],

    # values between 10 and 100 can be used, 2500 is for zero refractory time
    "I_pl_range":   [{neuron_parameter.I_pl : Ampere(1e-9*I)} for I in np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 2500])],

    # How many repetitions?
    # Each repetition will take about 1 minute per step!
    "repetitions":  1,

    # Set which calibrations you want to run
    "run_readout_shift": True,
    "run_V_reset":  True,
    "run_E_synx":   True,
    "run_E_syni":   True,
    "run_E_l_I_gl_fixed":  True,
    "run_E_l":      True,
    "run_V_t":      True,
    "run_I_gl":     False,
    "run_I_pl":     False,
    "run_V_syntcx": False,
    "run_V_syntci": False,
    "run_V_syntcx_psp_max": True,
    "run_V_syntci_psp_max": True,


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

    "V_reset_save_traces":  True,
    "V_t_save_traces":  True,

    # Where do you want to save the measurements (folder) and calibration data (backend)
    # Folders will be created if they do not exist already
    "folder":       folder,
    "backend":      os.path.join(folder, "backends"),

    # Wafer and HICANN coordinates
    "coord_wafer":  Wafer(),
    "coord_hicann": HICANNOnWafer(Enum(280)),


    # Here you can set the fixed parameters for each calibration.
    # base_parameters are set for all calibrations
    "base_parameters":   {neuron_parameter.E_l: Volt(E_l_target),
                          neuron_parameter.E_syni: Volt(E_syni_target),     # synapse
                          neuron_parameter.E_synx: Volt(E_synx_target),    # synapse
                          neuron_parameter.I_bexp: Ampere(1e-9*2500),       # turn off exp by setting this to 2500 and see I_rexp and V_bexp
                          neuron_parameter.I_convi: Ampere(1e-9*2500),   # bias current for synaptic input
                          neuron_parameter.I_convx: Ampere(1e-9*2500),   # bias current for synaptic input
                          neuron_parameter.I_fire: Ampere(0),       # adaptation term b
                          neuron_parameter.I_gladapt: Ampere(0),    # adaptation term
                          neuron_parameter.I_gl: Ampere(1e-9*1000),      # leakage conductance
                          neuron_parameter.I_intbbi: Ampere(1e-9*2000),  # integrator bias in synapse
                          neuron_parameter.I_intbbx: Ampere(1e-9*2000),  # integrator bias in synapse
                          neuron_parameter.I_pl: Ampere(1e-9*2000),      # tau_refrac
                          neuron_parameter.I_radapt: Ampere(1e-9*2500),  #
                          neuron_parameter.I_rexp: Ampere(1e-9*2500),     # exp-term: must be 2500 if I_bexp is 2500
                          neuron_parameter.I_spikeamp: Ampere(1e-9*2000),  #
                          neuron_parameter.V_exp: Volt(1.8),      # exponential term
                          neuron_parameter.V_syni: Volt(1.),    # inhibitory synaptic reversal potential
                          neuron_parameter.V_syntci: Volt(1.42),   # inhibitory synapse time constant
                          neuron_parameter.V_syntcx: Volt(1.42),   # excitatory synapse time constant
                          neuron_parameter.V_synx: Volt(1.),    # excitatory synaptic reversal potential
                          neuron_parameter.V_t: Volt(1.),       # recommended threshold, maximum is 1.1

                          shared_parameter.V_reset:       Volt(.5),   # * Neuron reset voltage
                          shared_parameter.int_op_bias:   DAC(1023),      # internal OP bias
                          shared_parameter.V_dllres:      DAC(200),       # DLL reset voltage
                          shared_parameter.V_bout:        Ampere(1e-9*750),   # * Global bias of neuron readout amps
                          shared_parameter.V_bexp:        Ampere(1e-9*2500),  # * exp-term: Must be set to max if I_bexp is 2500
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

    "readout_shift_parameters": {neuron_parameter.V_t: Volt(1.4),
                                 neuron_parameter.I_convx: Ampere(0),
                                 neuron_parameter.I_convi: Ampere(0),
                                 },

    "E_synx_parameters": {neuron_parameter.I_gl: Ampere(0),  # I_gl and I_convi MUST be set to 0
                          neuron_parameter.I_convx: Ampere(1e-9*2500),
                          neuron_parameter.I_convi: Ampere(0),
                          neuron_parameter.V_t: Volt(1.2),
                          shared_parameter.V_reset:  Volt(.2),
                          #neuron_parameter.V_syntcx: Volt(1.44, apply_calibration=True), # dummy
                          #neuron_parameter.V_syntci: Volt(1.44, apply_calibration=True), # dummy
                          },

    "E_syni_parameters": {neuron_parameter.I_gl: Ampere(0),  # I_gl and I_convx MUST be set to 0
                          neuron_parameter.I_convx: Ampere(0),
                          neuron_parameter.I_convi: Ampere(1e-9*2500),
                          neuron_parameter.V_t: Volt(1.2),
                          shared_parameter.V_reset:  Volt(.2),
                          },

    "E_l_I_gl_fixed_parameters": {neuron_parameter.V_t:        Volt(1.2),
                                  neuron_parameter.E_syni:     Volt(E_syni_target, apply_calibration=True),
                                  neuron_parameter.E_synx:     Volt(E_synx_target, apply_calibration=True),
                                  shared_parameter.V_reset:    Volt(.5, apply_calibration=True),
                                  },

    "E_l_parameters":       {neuron_parameter.V_t:        Volt(1.2),
                             neuron_parameter.E_syni:     Volt(E_syni_target, apply_calibration=True),
                             neuron_parameter.E_synx:     Volt(E_synx_target, apply_calibration=True),
                             neuron_parameter.I_gl:       Ampere(1e-9*1000, apply_calibration=True),
                             neuron_parameter.V_syntcx: Volt(1.44, apply_calibration=True),  # dummy
                             neuron_parameter.V_syntci: Volt(1.44, apply_calibration=True),  # dummy
                             shared_parameter.V_reset:    Volt(.5, apply_calibration=True),
                             },

    "V_syntcx_psp_max_parameters":       {neuron_parameter.V_t:        Volt(1.2),
                                          neuron_parameter.E_syni:     Volt(E_syni_target, apply_calibration=True),
                                          neuron_parameter.E_l:     Volt(E_l_target, apply_calibration=True),
                                          neuron_parameter.E_synx:     Volt(E_synx_target, apply_calibration=True),
                                          shared_parameter.V_reset:    Volt(.5, apply_calibration=True),
                                          },

    "V_syntci_psp_max_parameters":       {neuron_parameter.V_t:        Volt(1.2),
                                          neuron_parameter.E_syni:     Volt(E_syni_target, apply_calibration=True),
                                          neuron_parameter.E_l:     Volt(E_l_target, apply_calibration=True),
                                          neuron_parameter.E_synx:     Volt(E_synx_target, apply_calibration=True),
                                          shared_parameter.V_reset:    Volt(.5, apply_calibration=True),
                                          },

    "V_t_parameters":       {neuron_parameter.E_l:        Volt(1.2),
                             neuron_parameter.I_gl:       Ampere(1e-9*1500),
                             shared_parameter.V_reset:    Volt(.4),
                             neuron_parameter.I_convx: Ampere(0),
                             neuron_parameter.I_convi: Ampere(0),
                             },

    "V_reset_parameters":   {neuron_parameter.E_l:    Volt(1.3),
                             neuron_parameter.I_convx: Ampere(0),
                             neuron_parameter.I_convi: Ampere(0),
                             neuron_parameter.V_t:    Volt(.9),
                             neuron_parameter.I_gl:   Ampere(1e-9*1100),
                             neuron_parameter.I_pl:  Ampere(1e-9*20),
                             },

    "I_gl_parameters":       {neuron_parameter.E_l: Volt(E_l_target, apply_calibration=True),
                              neuron_parameter.V_t:        Volt(1.2),
                              neuron_parameter.E_syni:     Volt(E_syni_target, apply_calibration=True),
                              neuron_parameter.E_synx:     Volt(E_synx_target, apply_calibration=True),
                              shared_parameter.V_reset:    Volt(.2, apply_calibration=True),
                              },

    "I_pl_parameters":       {neuron_parameter.E_l:        Volt(1.2),
                              neuron_parameter.V_t:        Volt(.8),
                              neuron_parameter.E_syni:     Volt(E_syni_target, apply_calibration=True),
                              neuron_parameter.E_synx:     Volt(E_synx_target, apply_calibration=True),
                              shared_parameter.V_reset:    Volt(.5, apply_calibration=True),
                              },

    "V_syntcx_parameters":  {neuron_parameter.E_l:     Volt(E_l_target, apply_calibration=True),
                             neuron_parameter.E_syni:  Volt(E_syni_target, apply_calibration=True),
                             neuron_parameter.I_gl:    Ampere(1e-9*1000, apply_calibration=True),
                             neuron_parameter.E_synx:  Volt(E_synx_target, apply_calibration=True),
                             shared_parameter.V_gmax0: Volt(0.025),
                             shared_parameter.V_gmax1: Volt(0.025),
                             shared_parameter.V_gmax2: Volt(0.025),
                             shared_parameter.V_gmax3: Volt(0.025),
                             shared_parameter.V_reset: Volt(.4),
                             },

    # In which order should the parameters be calibrated?
    "parameter_order":      ['readout_shift',
                             shared_parameter.V_reset.name,
                             neuron_parameter.E_syni.name,
                             neuron_parameter.E_synx.name,
                             "E_l_I_gl_fixed",
                             neuron_parameter.E_l.name,
                             neuron_parameter.V_t.name,
                             "V_syntcx_psp_max",
                             "V_syntci_psp_max",
                             neuron_parameter.I_gl.name,
                             neuron_parameter.I_pl.name,
                             neuron_parameter.V_syntcx.name,
                             neuron_parameter.V_syntci.name,
                             neuron_parameter.E_l.name,  # V_syntc calibration might shift E_l, recalibrate
                             ],
}
