# -*- coding: utf-8 -*-

"""Configuration file for Simulator class."""

# Parameters variation
parameters_variation = False
variation = 1e-3

# Noise
noise = False
noise_std = 10e-3

# Spike time jitter
jitter = False
jitter_std = 1e-8

# a value of None means it must be set manually
# before running a simulation
_default_parameters = {"EL": None,
                       "gL": None,
                       "Vt": None,
                       "Vreset": None,
                       "C": 2.6,
                       "a": 1e-6,
                       "tw": 1.,
                       "b": 1e-6,
                       "dT": 10.*1e-6,
                       "Vexp": 1000.,
                       "gsynx": 1e-6,
                       "gsyni": 1e-6,
                       "tausynx": 10.,
                       "tausyni": 10.,
                       "tauref": 0.,
                       "Esynx": 1e-6,
                       "Esyni": 1e-6
                       }

_time = {"gL": 30e-6,
         "a": 200e-6,
         "Vexp": 200e-6,
         "b": 100e-6,
         "tauw_isi": 10e-6,
         "tauw_int": 50e-6,
         "tausyn_PSP": 100e-6
         }


def get_parameters(parameter):
    """Changes default parameters for specific parameter.

    Returns:
        parameters (dict)
    """
    parameters = dict(_default_parameters)

    if parameter in ("gL", "tausyn"):
        parameters["tw"] = 30.
    elif parameter in ("a", "b"):
        parameters["tw"] = 1.

    if parameter in ("gL", "tauw", "tauw_isi", "tauw_int", "tausyn", "tausyn_PSP"):
        parameters["dT"] = 10.*1e-6
    elif parameter in ("a", "b"):
        parameters["dT"] = 300.*1e-6

    if parameter == "Vexp":
        parameters["dT"] = 10.

    if parameter == "tausyn":
        parameters["gsynx"] = 100

    return parameters


def get_time(parameter):
    return _time[parameter]
