First attempt at an easy-to-use calibration tool.
Set your configuration in the parameters.py file, then execute run_calibration.py

Note:
- Only E_l, V_t and V_reset calibration work.
- V_reset cannot be calibrated per neuron, only per block of 128 neurons
It is planned to adjust the other voltage parameters for this deviation so that each neuron has the same dynamic range
