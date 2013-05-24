# -*- coding: utf-8 -*-

"""Hardware configuration options. Used in hardware interface."""

FPGA_PORT = 1701

# Voltage conversion
vCoeff = 1000  # FIXME: should be in calibration

current_default = 600  # Default current, DAC value

# sample times, maybe move into pycairo.config.adc?
sample_time_tw = 400
sample_time_dT = 400

# Floating gate parameters
res_fg = 1023
max_v = 1800  # mV
max_i = 2500  # nA

pll = 150  # PLL frequency
fg_pll = 100  # PLL frequency when writing floating gates
