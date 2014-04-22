import pyhalbe
from pyhalbe.Coordinate import NeuronOnHICANN, FGBlockOnHICANN, Enum
from pycake.experiment import Voltage, Current
import copy
import sys
import os

neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

folder = "/home/np001/temp/restructure/"

parameters = {
# Which neurons and blocks do you want to calibrate?
        "neurons": [NeuronOnHICANN(Enum(i)) for i in range(512)],
        "blocks":  [FGBlockOnHICANN(Enum(i)) for i in range(4)],

        # Set the ranges within which you want to calibrate
        "E_synx_range": range(650,950,100),    # 3 steps
        "E_syni_range": range(350,650,100),    # 3 steps
        "E_l_range":    range(500,800,100),      # 3 steps
        "V_t_range":    range(600,900,100),      # 3 steps
        "V_reset_range":range(400,700,100),      # 3 steps
        "I_gl_range":   range(250,300,100),     # 4 steps -> 19 steps

        # How far should the E_syn values be set around E_l
        "E_syni_dist":  -100,
        "E_synx_dist":  100,

        # How many repetitions? Each repetition will take about 1 minute per step!
        "repetitions":  1,
        
        # Set which calibrations you want to run
        "run_V_reset":  True,
        "run_E_synx":   False,
        "run_E_syni":   False,
        "run_E_l":      True,
        "run_V_t":      False,
        "run_I_gl":     False, # TODO g_l calibration is not yet implemented!
        "run_V_syntcx": False,
        "run_V_syntci": False,


        # Measurement runs twice by default: first to generate calibration data, and a second time to measure the success of calibration
        # Here you can turn either of these runs on or off
        "calibrate":    True,
        "measure":      True,

        # Overwrite old calibration data? This will not reset defect neurons!
        # Or even clear ALL calibration data before starting?
        "overwrite":    False,
        "clear":        True,

        # Set whether you want to keep traces or delete them after analysis
        "save_traces":  True,

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
        "coord_hicann": pyhalbe.Coordinate.HICANNOnWafer(pyhalbe.Coordinate.Enum(280)),

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
        "base_parameters":   {  neuron_parameter.E_l: Voltage(600),
                                neuron_parameter.E_syni: Voltage(500),     # synapse
                                neuron_parameter.E_synx: Voltage(700),    # synapse
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
                                neuron_parameter.I_spikeamp: Current(2000),#
                                neuron_parameter.V_exp: Voltage(1800),      # exponential term
                                neuron_parameter.V_syni: Voltage(1000),    # inhibitory synaptic reversal potential
                                neuron_parameter.V_syntci: Voltage(1375),   # inhibitory synapse time constant
                                neuron_parameter.V_syntcx: Voltage(1375),   # excitatory synapse time constant
                                neuron_parameter.V_synx: Voltage(1000),    # excitatory synaptic reversal potential
                                neuron_parameter.V_t: Voltage(1000),       # recommended threshold, maximum is 1100

                                shared_parameter.V_bexp: Current(2500), # exp-term: Must be set to max if I_bexp is 2500
                                shared_parameter.V_bout: Current(750),
                                shared_parameter.V_gmax0: Voltage(80),
                                shared_parameter.V_gmax1: Voltage(80),
                                shared_parameter.V_gmax2: Voltage(80),
                                shared_parameter.V_gmax3: Voltage(80),
                                },

        "E_synx_parameters":     {  neuron_parameter.I_gl: Current(0), # I_gl and I_convi MUST be set to 0
                                    neuron_parameter.I_convx: Current(2500),
                                    neuron_parameter.I_convi: Current(0),
                                    neuron_parameter.V_syntcx: Voltage(1800),
                                    neuron_parameter.V_syntci: Voltage(1800),
                                    neuron_parameter.V_t: Voltage(1200),
                                    shared_parameter.V_reset:  Voltage(200),
                                },

        "E_syni_parameters":     {  neuron_parameter.I_gl: Current(0), # I_gl and I_convx MUST be set to 0
                                    neuron_parameter.I_convx: Current(0),
                                    neuron_parameter.I_convi: Current(2500),
                                    neuron_parameter.V_syntci: Voltage(1800),
                                    neuron_parameter.V_syntcx: Voltage(1800),
                                    neuron_parameter.V_t: Voltage(1200),
                                    shared_parameter.V_reset:  Voltage(200),
                                },

        "E_l_parameters":       {   neuron_parameter.V_t:        Voltage(1200),
                                    neuron_parameter.I_gl:       Current(1500),
                                    shared_parameter.V_reset:    Voltage(300, apply_calibration = True),
                                },

        "V_t_parameters":       {   neuron_parameter.E_l:        Voltage(1000),
                                    neuron_parameter.I_gl:       Current(1000),
                                    shared_parameter.V_reset:    Voltage(400),
                                },

        "V_reset_parameters":   {   neuron_parameter.E_l:    Voltage(1100),
                                    neuron_parameter.V_t:    Voltage(800),
                                    neuron_parameter.I_gl:   Current(1000),
                                    neuron_parameter.I_pl:  Current(50),
                                },

        "I_gl_parameters":       {   neuron_parameter.E_l:        Voltage(600),
                                    neuron_parameter.V_t:        Voltage(1200),
                                    shared_parameter.V_reset:    Voltage(200),
                                },

        # In which order should the parameters be calibrated?
        "parameter_order":      [   shared_parameter.V_reset,
                                    neuron_parameter.E_syni,
                                    neuron_parameter.E_synx,
                                    neuron_parameter.E_l,
                                    neuron_parameter.V_t,
                                    neuron_parameter.I_gl,
                                    neuron_parameter.V_syntcx,
                                    neuron_parameter.V_syntci
                                ],
}


