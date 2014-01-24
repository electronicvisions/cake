import pyhalbe
from pycake.experiment import Voltage, Current
import copy
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

# Set coarse to True if you only want to sweep very few data points
# This should only be used for debugging (in order to see if everything works
coarse = True 

parameters = {
        # Set the ranges within which you want to calibrate
        "E_synx_range": range(650,1050,25),    # 16 steps
        "E_syni_range": range(350,750,25),    # 16 steps
        "E_l_range":    range(500,800,25),      # 12 steps
        "V_t_range":    range(600,900,25),      # 12 steps
        "V_reset_range":range(400,700,25),      # 12 steps
        "I_gl_range":   range(50,2500,50),     # 50 steps

        # How far should the E_syn values be set around E_l
        "E_syni_dist":  -100,
        "E_synx_dist":  100,

        # How many repetitions? Each repetition will take about 1 minute per step!
        "repetitions":  4,
        
        # Set which calibrations you want to run
        "run_E_synx":  True,
        "run_E_syni":  False,
        "run_E_l":     False,
        "run_V_t":     False,
        "run_V_reset": False,
        "run_I_gl":    False, # TODO g_l calibration is not yet implemented!

        # Measurement runs twice by default: first to generate calibration data, and a second time to measure the success of calibration
        # Here you can turn either of these runs on or off
        "calibrate":    True,
        "measure":      True,

        # Overwrite old calibration data? This will not reset defect neurons!
        # Or even clear ALL calibration data before starting?
        "overwrite":    True,
        "clear":        True,

        # save_results will save all the measurements in a folder specified below.
        # This has nothing to do with the calibration data which is stored anyway!
        # You can also save all the traces for debugging purposes. Note that this takes a lot of space (100 MB per repetition)
        "save_results": True,
        "save_traces":  False,

        # If you save al your measurements, each folder will have a description file. The following parameters let you specify additional info to be stored.
        "E_synx_description":   "",
        "E_syni_description":   "",
        "E_l_description":      "(E_syni,E_synx) = (100,100). Esyn CALIBRATED and set after E_l calibration.",  
        "V_t_description":      "",  
        "V_reset_description":  "",
        "I_gl_description":     "Large range",

        # Where do you want to save the measurements (folder) and calibration data (backend_c for calibtic, backend_r for redman)?
        # Folders will be created if they do not exist already
        "folder":       "/home/np001/temp/I_gl_100/",
        "backend_c":    "/home/np001/temp/I_gl_100/backends/",
        "backend_r":    "/home/np001/temp/I_gl_100/backends/",
        
        # Wafer and HICANN coordinates
        "coord_wafer":  pyhalbe.Coordinate.Wafer(),
        "coord_hicann": pyhalbe.Coordinate.HICANNOnWafer(pyhalbe.Coordinate.Enum(280)),


        # ADVANCED STUFF:
        # Set the fix parameters you wish to have while calibrating
        "calibration_params": {        
            # Global parameters are set for ALL calibrations.
            "global_params":    {   neuron_parameter.I_gl: Current(1000),
                                },

            "global_shared":    {   shared_parameter.V_reset: Voltage(500),
                                },
            
            # Set other values for different experiments. These values overwrite global values.
            "E_syn_params":     {   neuron_parameter.V_t:    Voltage(1200),
                                    neuron_parameter.I_gl:   Current(0),
                                },
            "E_syn_shared":     {   shared_parameter.V_reset:  Voltage(1200),
                                },

            "E_l_params":       {   neuron_parameter.V_t:        Voltage(1200),
                                    neuron_parameter.I_gl:       Current(1000),
                                },  
            "E_l_shared":       {   shared_parameter.V_reset:      Voltage(300),
                                },

            "V_t_params":       {   neuron_parameter.E_l:        Voltage(1000),
                                    neuron_parameter.I_gl:       Current(1000),
                                },
            "V_t_shared":       {   shared_parameter.V_reset:    Voltage(400),
                                },

            "V_reset_params":   {   neuron_parameter.E_l:    Voltage(1100),
                                    neuron_parameter.V_t:    Voltage(800),
                                    neuron_parameter.I_gl:   Current(1000),
                                },
            "V_reset_shared":   {
                                },

            "g_l_params":       {   neuron_parameter.E_l:        Voltage(700),
                                    neuron_parameter.V_t:        Voltage(600),
                                },
            "g_l_shared":       {   shared_parameter.V_reset:    Voltage(450),
                            },
        },
        }

test_params = copy.deepcopy(parameters["calibration_params"])
for exp in test_params:
    for par in test_params[exp]:
        test_params[exp][par].apply_calibration = True
parameters['test_params'] = test_params


if coarse:
    parameters.update({
           "E_synx_range": range(650,1050,100),    # 4 steps
           "E_syni_range": range(350,750,100),     # 4 steps
           "E_l_range":    range(500,800,100),    # 3 steps
           "V_t_range":    range(550,750,50),     # 3 steps
           "V_reset_range":range(300,600,100),    # 3 steps
           "I_gl_range":   range(100,200,100),
           "E_synx_description":   parameters["E_synx_description"] + " Coarse.",
           "E_syni_description":   parameters["E_syni_description"] + " Coarse.",
           "E_l_description":      parameters["E_l_description"] + " Coarse.",
           "V_t_description":      parameters["V_t_description"] + " Coarse.",
           "V_reset_description":  parameters["V_reset_description"] + " Coarse.",
           "I_gl_description":     parameters["I_gl_description"] + " Coarse.",
           "repetitions":     1,
           })


