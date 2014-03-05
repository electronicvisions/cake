import pyhalbe
from pycake.experiment import Voltage, Current
import copy
import os
import sys

neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

folder = "/tmp/calibration"

parameters = {
        # Set the ranges within which you want to calibrate
        "V_reset_range":range(200,800,25),      # 12 steps
        "E_synx_range": range(650,1100,25),    # 16 steps
        "E_syni_range": range(350,850,25),    # 16 steps
        "E_l_range":    range(500,900,25),      # 12 steps
        "V_t_range":    range(600,900,25),      # 12 steps
        "I_gl_range":   range(100,150,50),     # 50 steps

        # How far should the E_syn values be set around E_l
        "E_syni_dist": -150,
        "E_synx_dist":  150,

        # How many repetitions? Each repetition will take about 1 minute per step!
        "repetitions":  3,

        # Set which calibrations you want to run
        "run_V_reset": True,
        "run_E_synx":  True,
        "run_E_syni":  True,
        "run_E_l":     True,
        "run_V_t":     True,
        "run_I_gl":    True,

        # Measurement runs twice by default: first to generate calibration data, and a second time to measure the success of calibration
        # Here you can turn either of these runs on or off
        "calibrate":    True,
        "measure":      True,

        # Overwrite old calibration data? This will not reset defect neurons!
        "overwrite":    True,
        # Or even clear ALL calibration data before starting?
        "clear":        False,

        # save_results will save all the measurements in a folder specified below.
        # This has nothing to do with the calibration data which is stored anyway!
        # You can also save all the traces for debugging purposes. Note that this takes a lot of space (20 MB per repetition)
        "save_results": True,
        "save_traces":  True,

        # If you save al your measurements, each folder will have a description file. The following parameters let you specify additional info to be stored.
        "V_reset_description":  "V_reset and readout shift calibration.",
        "E_synx_description":   "E_synx calibration.",
        "E_syni_description":   "E_syni calibration.",
        "E_l_description":      "E_l calibration.",
        "V_t_description":      "V_t calibration.",
        "I_gl_description":     "I_gl calibration.",

        # Where do you want to save the measurements (folder) and calibration data (backend_c for calibtic, backend_r for redman)?
        # Folders will be created if they do not exist already
        "folder":       folder,
        "backend_c":    os.path.join(folder, "backends"),
        "backend_r":    os.path.join(folder, "backends"),

        # Wafer and HICANN coordinates
        "coord_wafer":  pyhalbe.Coordinate.Wafer(),
        "coord_hicann": pyhalbe.Coordinate.HICANNOnWafer(pyhalbe.Coordinate.Enum(280)),

        # Maximum tries in case an experiment should fail
        "max_tries":    3,


        # If you want to load previously measured traces instead of
        # measuring new ones, specify the experiment folders here
        "E_synx_traces":   None,
        "E_syni_traces":   None,
        "E_l_traces":      None,
        "V_t_traces":      None,
        "V_reset_traces":  None,
        "I_gl_traces":     None,


        # Here you can set the fixed parameters for each calibration.
        # base_parameters are set for all calibrations 
        "base_parameters":   {  neuron_parameter.E_l: Voltage(600),
                                neuron_parameter.E_syni: Voltage(500),     # synapse
                                neuron_parameter.E_synx: Voltage(700),    # synapse
                                neuron_parameter.I_bexp: Current(2500),       # exponential term set to 0
                                neuron_parameter.I_convi: Current(2500),   # bias current for synaptic input
                                neuron_parameter.I_convx: Current(2500),   # bias current for synaptic input
                                neuron_parameter.I_fire: Current(0),       # adaptation term b
                                neuron_parameter.I_gladapt: Current(0),    # adaptation term
                                neuron_parameter.I_gl: Current(1000),      # leakage conductance
                                neuron_parameter.I_intbbi: Current(2000),  # integrator bias in synapse
                                neuron_parameter.I_intbbx: Current(2000),  # integrator bias in synapse
                                neuron_parameter.I_pl: Current(2000),      # tau_refrac
                                neuron_parameter.I_radapt: Current(2500),  #
                                neuron_parameter.I_rexp: Current(2500),     # something about the strength of exp term
                                neuron_parameter.I_spikeamp: Current(2000),#
                                neuron_parameter.V_exp: Voltage(1800),      # exponential term
                                neuron_parameter.V_syni: Voltage(1000),    # technical parameter that drives the integrator 
                                neuron_parameter.V_syntci: Voltage(1375),   # inhibitory synapse time constant
                                neuron_parameter.V_syntcx: Voltage(1375),   # excitatory synapse time constant
                                neuron_parameter.V_synx: Voltage(1000),    # technical parameter that drives the integrator 
                                neuron_parameter.V_t: Voltage(1000),       # recommended threshold, maximum is 1100

                                shared_parameter.V_bexp: Current(2500),
                                shared_parameter.V_bout: Current(750),
                                },

        "E_synx_parameters":     {  neuron_parameter.I_gl: Current(0),          # I_gl and I_convi MUST be set to 0
                                    neuron_parameter.I_convi: Current(0),
                                    neuron_parameter.V_syntcx: Voltage(1800),
                                    neuron_parameter.V_syntci: Voltage(1800),
                                    neuron_parameter.V_t:       Voltage(1250),
                                    shared_parameter.V_reset:  Voltage(100),
                                },

        "E_syni_parameters":     {  neuron_parameter.I_gl: Current(0),
                                    neuron_parameter.I_convx: Current(0),
                                    neuron_parameter.V_syntci: Voltage(1800),
                                    neuron_parameter.V_syntcx: Voltage(1800),
                                    neuron_parameter.V_t:       Voltage(1250),
                                    shared_parameter.V_reset:  Voltage(100),
                                },

        "E_l_parameters":       {   neuron_parameter.V_t:        Voltage(1200),
                                    neuron_parameter.I_gl:       Current(1000),
                                    shared_parameter.V_reset:      Voltage(300),
                                },

        "V_t_parameters":       {   neuron_parameter.E_l:        Voltage(1050),
                                    neuron_parameter.I_gl:       Current(1000),
                                    shared_parameter.V_reset:    Voltage(400),
                                },

        "V_reset_parameters":   {   neuron_parameter.E_l:    Voltage(1100),
                                    neuron_parameter.V_t:    Voltage(900),
                                    neuron_parameter.I_gl:   Current(1000),
                                },

        "I_gl_parameters":       {  neuron_parameter.E_l:        Voltage(600),
                                    neuron_parameter.V_t:        Voltage(1200),
                                    shared_parameter.V_reset:    Voltage(200),
                                },
}


