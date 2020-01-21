import pyhalbe
from pyhalco_common import Enum
from pyhalco_hicann_v2 import HICANNOnWafer, NeuronOnHICANN, FGBlockOnHICANN
from pycake.helpers.units import Voltage, Current
from pycake.helpers.units import linspace_voltage, linspace_current

neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

from fastcalibration_parameters import parameters

from fastcalibration_parameters import E_l_target
from fastcalibration_parameters import E_syni_target
from fastcalibration_parameters import E_synx_target

eval_parameters = {
# Which neurons and blocks do you want to calibrate?
        "filename_prefix":  "mashup",
        "neurons": [NeuronOnHICANN(Enum(i)) for i in range(512)],
        "blocks":  [FGBlockOnHICANN(Enum(i)) for i in range(4)],

        # For the evaluation of the calibration, only one point, except for V_t, is targeted
        "E_l_range":      [{neuron_parameter.E_l : Voltage(E_l_target)}],

        # How many repetitions?
        # Each repetition will take about 1 minute per step!
        "repetitions":  1,

        "wafer_cfg" : "wafer.xml",

        # has to match wafer_cfg
        "coord_hicann": HICANNOnWafer(Enum(276)),

        # Set which calibrations you want to run
        "run_V_reset":  False,
        "run_E_synx":   False,
        "run_E_syni":   False,
        "run_E_l":      True,
        "run_V_t":      False,
        "run_I_gl":     False,
        "run_I_pl":     False,
        "run_V_syntcx": False,
        "run_V_syntci": False,
        "run_V_syntci_psp_max": False,
        "run_V_syntcx_psp_max": False,
        "run_E_l_I_gl_fixed":  False,

        # Measurement runs twice by default: first to generate calibration data, and a second time to measure the success of calibration
        # Here you can turn either of these runs on or off
        "calibrate":    False,
        "measure":      True,

}

parameters.update(eval_parameters)
