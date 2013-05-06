"""Hardware configuration options. Used in hardware interface."""

# Voltage conversion
vCoeff = 1000 # FIXME: should be in calibration

current_default = 600 # Default current, DAC value

# sample times, maybe move into pycairo.config.adc?
sample_time_tw = 400
sample_time_dT = 400
