# -*- coding: utf-8 -*-

"""Polynom coefficients of the ideal AdEx transformation."""

ident = [1, 0]

# copied from scaledtsim process="180" and verified from Marco's thesis
# TODO add reference to table in Marco's thesis
IDEAL_TRAFO_ADEX = {  # Don't change
    "Vreset":  ident,
    # From transistor-level simulations
    "EL":      [1.02, -8.58],
    "Vt":      [0.998, -3.55],
    "Esynx":   [1.02, -8.58],
    "Esyni":   [1.02, -8.58],
    "gL":      [5.52e-5, 0.24, 0.89],
    "tauref":  [0.025, -0.0004],  # INVERSE
    "a":       [4.93e-5, 0.26, -0.66],
    "b":       [-0.14, 45, 54.75],
    "tw":      [-4.4e-6, 0.00032, -0.0005],  # INVERSE
    "dT":      [9.2385890861, 66.3846854343, -94.2540733183],
    "Vexp":    [0.371990571979, 100.290157129],  # for dT = 8 mV
    "tausynx": [-3.94, 37, 1382],
    "tausyni": [-3.94, 37, 1382],
}
