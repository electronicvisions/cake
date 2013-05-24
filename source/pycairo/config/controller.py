debug_mode = False
debug_print_results = True
verbose = True  # print what is going on

# Calibration range for each parameter: min, max, number of values, number of repetitions
parameter_ranges = {
    "EL": {"min": 300, "max": 800, "pts": 6, "reps": 3},
    "Vreset": {"min": 400, "max": 600, "pts": 3, "reps": 2},
    "Vt": {"min": 600, "max": 800, "pts": 3, "reps": 2},
    "gL": {"min": 100, "max": 600, "pts": 4, "reps": 4},
    "tauref": {"min": 10, "max": 100, "pts": 4, "reps": 4},
    "dT": {"min": 50, "max": 400, "pts": 3, "reps": 3},
    "Vexp": {"min": 100, "max": 400, "pts": 3, "reps": 3},
    "b": {"min": 50, "max": 2000, "pts": 3, "reps": 3},
    "tausynx": {"min": 1300, "max": 1500, "pts": 4, "reps": 4},
    "tausyni": {"min": 1300, "max": 1500, "pts": 3, "reps": 2},
    "tw": {"min": 50, "max": 2000, "pts": 3, "reps": 3},
    "a": {"min": 50, "max": 400, "pts": 5, "reps": 5}
}

# Calibration default values
parameter_default = {
    "gL": 1000,  # nA
    "EL": 600,  # mV
    "a": 2000,  # nA
    "tauw": 1000,  # nA
    "Vexp": 650,  # mV
    "dT": 10,  # mV
    "Esynx": 1300,  # mV
    "Esyni": 200,  # mV
    "current": 200  # nA
}

parameter_max = {
    "Vt": 1700,  # mV
    "EL": 1100  # mV
}

parameter_IF = {
    "EL": 800,  # mV
    "Vreset": 500,  # mV
    "Vt": 700,  # mV
    "gL": 1000  # nA
}

parameter_special = {  # FIXME add comment, what are these?!
    "gL": 1000,
    "expAct": 2000
}
