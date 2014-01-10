import pyhalbe

coarse = True      # Set this to True if you only want to sweep a few data points.

parameters = {
        "E_syni_dist":  -100,       # How far should the E_syn values be set around E_l?
        "E_synx_dist":  100,
    
        "E_l_range":    range(500,800,25),      # 12 steps
        "V_t_range":    range(600,900,25),      # 12 steps
        "V_reset_range":range(400,700,25),      # 12 steps
        "I_gl_range":   range(50,2500,50),     # 50 steps
        "E_synx_range": range(650,1050,25),    # 16 steps
        "E_syni_range": range(350,750,25),    # 16 steps

        
        "repetitions":  5,          # How many repetitions per step? Each repetition takes about 1 minute on the vertical setup
        
        "run_E_l":     False,       # Do E_l calibration?
        "run_V_t":     False,       # Do V_t calibration?
        "run_V_reset": False,       # Do V_reset calibration?
        "run_I_gl":    False,        # TODO implement proper I_gl calibration. Until now, only measurement is done (without calibration)
        "run_E_synx":  True,        # Do E_synx calibration?
        "run_E_syni":  True,        # Do E_syni calibration?

        "calibrate":    True,       # Run the calibration? Turn this off if only a measurement is wanted
        "measure":      True,       # Measure after calibration? Should not be False if you calibrate, because every calibration needs to be checked!

        "dry_run":      False,       # TODO implement this
        "save_results": True,        # Save measurement results in addition to fit parameters?
        "save_traces":  False,       # Save all traces? Note that traces for all neurons take about 100 MB per measurement! Total disk space would be 100 MB *steps*repetitions

        "E_l_description":      "(E_syni,E_synx) = (100,100)",  # Suffixes to the measurements to give further info on what this measurement is about
        "V_t_description":      "",                             # The description will later be something like "E_l calibration: <this description>" etc.
        "V_reset_description":  "",
        "I_gl_description":     "Large range",
        "E_synx_description":   "",
        "E_syni_description":   "",

        "folder":       "/home/np001/temp/E_syn_simple/",  # Where do you want the measurement folders to be stored. Make sure the folder exists before running experiment
        "backend_c":    "/home/np001/backends/esyn_100_100/",   # Location of the calibitic backend. Make sure the folder exists before running experiment
        "backend_r":    "/home/np001/backends/esyn_100_100/",   # Location of the redman backend. Make sure the folder exists before running experiment
        
        "coord_wafer":  pyhalbe.Coordinate.Wafer(),     # Coordinate of the wafer
        "coord_hicann": pyhalbe.Coordinate.HICANNOnWafer(pyhalbe.Coordinate.Enum(280)), # Coordinate of HICANN on wafer
        }







if coarse:
    parameters.update({
           "E_l_range":    range(500,800,100),
           "V_t_range":    range(550,750,50),
           "V_reset_range":range(300,600,100),
           "I_gl_range":   range(600,1400,100),
           "E_synx_range": range(650,1050,50),    # 8 steps
           "E_syni_range": range(350,750,50),     # 8 steps
           "E_l_description":      parameters["E_l_description"] + " Coarse.",
           "V_t_description":      parameters["V_t_description"] + " Coarse.",
           "V_reset_description":  parameters["V_reset_description"] + " Coarse.",
           "I_gl_description":     parameters["I_gl_description"] + " Coarse.",
           "E_synx_description":   parameters["E_synx_description"] + " Coarse.",
           "E_syni_description":   parameters["E_syni_description"] + " Coarse.",
           "repetitions":     1,
           })


