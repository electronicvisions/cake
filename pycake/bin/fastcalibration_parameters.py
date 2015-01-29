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

E_l_target = 700  # FIXME: propagate to E_l_I_gl_fixed calibrator
E_syn_distance = 100

E_syni_target = E_l_target - E_syn_distance
E_synx_target = E_l_target + E_syn_distance

parameters = {
    # Which neurons and blocks do you want to calibrate?
    "filename_prefix": "",
    "neurons": [NeuronOnHICANN(Enum(i)) for i in range(512)],
    "blocks":  [FGBlockOnHICANN(Enum(i)) for i in range(4)],

    # Set the ranges within which you want to calibrate
    "V_reset_range":  [{shared_parameter.V_reset : v} for v in linspace_voltage(600, 800, 5)],
    "E_syni_range":   [{neuron_parameter.E_syni : v} for v in linspace_voltage(550, 850, 5)],
    "E_synx_range":   [{neuron_parameter.E_synx : v} for v in linspace_voltage(650, 950, 5)],
    "E_l_range":      [{neuron_parameter.E_l : v} for v in linspace_voltage(550, 850, 6)],
    "V_t_range":      [{neuron_parameter.V_t : v} for v in linspace_voltage(600, 900, 4)],
    "E_l_I_gl_fixed_range": pproduct((neuron_parameter.E_l, neuron_parameter.I_gl),
                                     (linspace_voltage(550, 850, 2), [DAC(v) for v in (40, 80, 120, 240, 480, 960)])),
    "I_gl_range":     [{neuron_parameter.I_gl : v} for v in linspace_current(100, 800, 8)],
    "V_syntcx_range": [{neuron_parameter.V_syntcx : v} for v in linspace_voltage(1200, 1660, 20)],
    "V_syntcx_psp_max_range": [{neuron_parameter.V_syntcx : v} for v in linspace_voltage(1350, 1700, 10)],
    "V_syntci_psp_max_range": [{neuron_parameter.V_syntci : v} for v in linspace_voltage(1350, 1700, 10)],

    # values between 10 and 100 can be used, 2500 is for zero refractory time
    "I_pl_range":   [{neuron_parameter.I_pl : Current(v)} for v in np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 2500])],

    # How many repetitions?
    # Each repetition will take about 1 minute per step!
    "repetitions":  1,

    # Set which calibrations you want to run
    "run_V_reset":  True,
    "run_E_synx":   True,
    "run_E_syni":   True,
    "run_E_l_I_gl_fixed":  True,
    "run_E_l":      True,
    "run_V_t":      True,
    "run_I_gl":     False,
    "run_I_pl":     True,
    "run_V_syntcx": False,
    "run_V_syntci": False,
    "run_V_syntcx_psp_max": True,
    "run_V_syntci_psp_max": True,
    "run_Spikes":   False,


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
    "V_reset_save_traces": True,
    "V_t_save_traces": True,

    ## If you save your measurements, each folder will have a description file. The following parameters let you specify additional info to be stored.
    #"E_synx_description":   "E_synx calibration.",
    #"E_syni_description":   "E_syni calibration",
    #"E_l_description":      "E_l calibration",
    #"V_t_description":      "V_t calibration",
    #"V_reset_description":  "V_reset calibration",
    #"I_gl_description":     "g_l measurement",

    # Where do you want to save the measurements (folder) and calibration data (backend_c for calibtic, backend_r for redman)?
    # Folders will be created if they do not exist already
    "folder":       folder,
    "backend_c":    os.path.join(folder, "backends"),
    "backend_r":    os.path.join(folder, "backends"),

    # Wafer and HICANN coordinates
    "coord_wafer":  pyhalbe.Coordinate.Wafer(),
    "coord_hicann": pyhalbe.Coordinate.HICANNOnWafer(pyhalbe.Coordinate.Enum(276)),

    ## Maximum tries in case an experiment should fail
    #"max_tries":    3,


    ## If you want to load previously measured traces instead of
    ## measuring new ones, specify the experiment folders here
    #"E_synx_traces":   None,
    #"E_syni_traces":   None,
    #"E_l_traces":      None,
    #"V_t_traces":      None,
    #"V_reset_traces":  None,
    #"I_gl_traces":     None,


    # Here you can set the fixed parameters for each calibration.
    # base_parameters are set for all calibrations
    "base_parameters":   {neuron_parameter.E_l: Voltage(E_l_target),
                          neuron_parameter.E_syni: Voltage(E_syni_target),     # synapse
                          neuron_parameter.E_synx: Voltage(E_synx_target),    # synapse
                          neuron_parameter.I_bexp: Current(2500),       # turn off exp by setting this to 2500 and see I_rexp and V_bexp
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

    "E_synx_parameters": {neuron_parameter.I_gl: Current(0),  # I_gl and I_convi MUST be set to 0
                          neuron_parameter.I_convx: Current(2500),
                          neuron_parameter.I_convi: Current(0),
                          neuron_parameter.V_t: Voltage(1200),
                          shared_parameter.V_reset:  Voltage(200),
                          #neuron_parameter.V_syntcx: Voltage(1440, apply_calibration=True), # dummy
                          #neuron_parameter.V_syntci: Voltage(1440, apply_calibration=True), # dummy
                          },

    "E_syni_parameters": {neuron_parameter.I_gl: Current(0),  # I_gl and I_convx MUST be set to 0
                          neuron_parameter.I_convx: Current(0),
                          neuron_parameter.I_convi: Current(2500),
                          neuron_parameter.V_t: Voltage(1200),
                          shared_parameter.V_reset:  Voltage(200),
                          },

    "E_l_I_gl_fixed_parameters": {neuron_parameter.V_t:        Voltage(1200),
                                  neuron_parameter.E_syni:     Voltage(E_syni_target, apply_calibration=True),
                                  neuron_parameter.E_synx:     Voltage(E_synx_target, apply_calibration=True),
                                  shared_parameter.V_reset:    Voltage(500, apply_calibration=True),
                                  },

    "E_l_parameters":       {neuron_parameter.V_t:        Voltage(1200),
                             neuron_parameter.E_syni:     Voltage(E_syni_target, apply_calibration=True),
                             neuron_parameter.E_synx:     Voltage(E_synx_target, apply_calibration=True),
                             neuron_parameter.I_gl:       Current(1000, apply_calibration=True),
                             neuron_parameter.V_syntcx: Voltage(1440, apply_calibration=True),  # dummy
                             neuron_parameter.V_syntci: Voltage(1440, apply_calibration=True),  # dummy
                             shared_parameter.V_reset:    Voltage(500, apply_calibration=True),
                             },

    "V_syntcx_psp_max_parameters":       {neuron_parameter.V_t:        Voltage(1200),
                                          neuron_parameter.E_syni:     Voltage(E_syni_target, apply_calibration=True),
                                          neuron_parameter.E_l:     Voltage(E_l_target, apply_calibration=True),
                                          neuron_parameter.E_synx:     Voltage(E_synx_target, apply_calibration=True),
                                          shared_parameter.V_reset:    Voltage(500, apply_calibration=True),
                                          },

    "V_syntci_psp_max_parameters":       {neuron_parameter.V_t:        Voltage(1200),
                                          neuron_parameter.E_syni:     Voltage(E_syni_target, apply_calibration=True),
                                          neuron_parameter.E_l:     Voltage(E_l_target, apply_calibration=True),
                                          neuron_parameter.E_synx:     Voltage(E_synx_target, apply_calibration=True),
                                          shared_parameter.V_reset:    Voltage(500, apply_calibration=True),
                                          },

    "V_t_parameters":       {neuron_parameter.E_l:        Voltage(1200),
                             neuron_parameter.I_gl:       Current(1500),
                             shared_parameter.V_reset:    Voltage(400),
                             neuron_parameter.I_convx: Current(0),
                             neuron_parameter.I_convi: Current(0),
                             },

    "V_reset_parameters":   {neuron_parameter.E_l:    Voltage(1300),
                             neuron_parameter.I_convx: Current(0),
                             neuron_parameter.I_convi: Current(0),
                             neuron_parameter.V_t:    Voltage(900),
                             neuron_parameter.I_gl:   Current(1100),
                             neuron_parameter.I_pl:  Current(20),
                             },

    "I_gl_parameters":       {neuron_parameter.E_l: Voltage(E_l_target, apply_calibration=True),
                              neuron_parameter.V_t:        Voltage(1200),
                              neuron_parameter.E_syni:     Voltage(E_syni_target, apply_calibration=True),
                              neuron_parameter.E_synx:     Voltage(E_synx_target, apply_calibration=True),
                              shared_parameter.V_reset:    Voltage(200, apply_calibration=True),
                              },

    "I_pl_parameters":       {neuron_parameter.E_l:        Voltage(1200),
                              neuron_parameter.V_t:        Voltage(800),
                              neuron_parameter.E_syni:     Voltage(E_syni_target, apply_calibration=True),
                              neuron_parameter.E_synx:     Voltage(E_synx_target, apply_calibration=True),
                              shared_parameter.V_reset:    Voltage(500, apply_calibration=True),
                              },

    "V_syntcx_parameters":  {neuron_parameter.E_l:     Voltage(E_l_target, apply_calibration=True),
                             neuron_parameter.E_syni:  Voltage(E_syni_target, apply_calibration=True),
                             neuron_parameter.I_gl:    Current(1000, apply_calibration=True),
                             neuron_parameter.E_synx:  Voltage(E_synx_target, apply_calibration=True),
                             shared_parameter.V_gmax0: Voltage(25),
                             shared_parameter.V_gmax1: Voltage(25),
                             shared_parameter.V_gmax2: Voltage(25),
                             shared_parameter.V_gmax3: Voltage(25),
                             shared_parameter.V_reset: Voltage(400),
                             },

    # In which order should the parameters be calibrated?
    "parameter_order":      [shared_parameter.V_reset.name,
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
                             "Spikes"
                             ],
}
