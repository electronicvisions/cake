"""Evaluate calibration in MonteCarlo transistor level simulation"""

from Coordinate import NeuronOnHICANN, FGBlockOnHICANN, Enum
from pycake.helpers.units import Voltage, Current
from pycake.helpers.units import linspace_voltage

from pyhalbe.HICANN import neuron_parameter
from pyhalbe.HICANN import shared_parameter

# original calibration config
from sim_denmem_parameters import parameters


eval_parameters = {
    "filename_prefix": "eval",
    "neurons": [NeuronOnHICANN(Enum(i)) for i in range(512)],
    "blocks": [FGBlockOnHICANN(Enum(i)) for i in range(4)],

    # For the evaluation of the calibration, sometimes only one point is
    # targeted, usually 3
    "V_reset_range": [{shared_parameter.V_reset : Voltage(500)}],
    "E_l_range": [{neuron_parameter.E_l : Voltage(v)} for v in [650, 700, 750]],
    "V_t_range": [{neuron_parameter.V_t : v} for v in linspace_voltage(700, 1100, 3)],
    "I_gl_charging_range": [{neuron_parameter.I_gl : Current(c)} for c in [500, 700]],
    "I_pl_range": [{neuron_parameter.I_pl : Current(c)} for c in [10, 30, 50, 70, 90, 2500]],

    # How many repetitions?
    # Each repetition will take about 1 minute per step!
    "repetitions": 1,

    # Set which evaluations you want to run
    "run_V_reset": True,
    "run_V_t": True,
    "run_E_syni": True,
    "run_E_synx": False,  # 2015-03-11: not calibrated
    "run_V_convoffx": False,  # constant value
    "run_V_convoffi": False,  # constant value
    "run_E_l": True,
    "run_I_pl": True,
    "run_I_gl_charging": True,
    "run_V_syntcx": False,  # no calibration
    "run_V_syntci": False,  # no calibration

    # Do not calibrate, just measure for evaluation
    "calibrate": False,
    "measure": True,
}

parameters.update(eval_parameters)
