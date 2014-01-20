# -*- coding: utf-8 -*-

"""Hardware configuration options and default hardware parameters used in calibration.

Other default hardware parameters can be found in Calibtic
and in HALbe (halbe/hal/HMFUtil.cpp).

Maybe we can get rid of the parameters in this file and use HALbe/Calibtic only.
"""


FPGA_PORT = 1701

current_default = 600  # Default current, DAC value

# sample times, maybe move into pycake.config.adc?
sample_time_tw = 400
sample_time_dT = 400

# Floating gate parameters
res_fg = 1023
max_v = 1800  # mV
max_i = 2500  # nA

pll = 150  # PLL frequency
fg_pll = 100  # PLL frequency when writing floating gates


_HW_parameters = {
    "EL": 1000,  # mV
    "Vt": 1000,  # mV
    "Vexp": 536,  # mV
    "Vreset": 500,  # mV
    "Esynx": 900,  # mV
    "Esyni": 900,  # mV
    "Vsynx": 0,  # mV
    "Vsyni": 0,  # mV
    "tausynx": 900,  # nA
    "tausyni": 900,  # nA
    "Ibexp": 0,  # nA
    "b": 0,  # nA
    "a": 0,  # nA
    "gL": 400,  # nA
    "tauref": 2000,  # nA
    "tw": 2000,  # nA
    "gsynx": 0.0,  # nA
    "gsyni": 0.0,  # nA
    "Iintbbx": 0,  # nA
    "Iintbbi": 0,  # nA
    "dT": 750,  # nA
    "Ispikeamp": 2000  # nA
}

V_gmax = 800
_Global_HW_parameters = {
    # these are floating gate values, except for V_reset
    "I_breset": 1023,
    "I_bstim": 1023,
    "int_op_bias": 1023,
    "V_bexp": 1023,  # only right
    "V_bout": 1023,  # only left
    "V_br": 0,
    "V_bstdf": 0,
    "V_ccas": 800,
    "V_clra": 0,     # only left
    "V_clrc": 0,     # only right
    "V_dep": 0,
    "V_dllres": 400,
    "V_dtc": 0,
    "V_fac": 0,
    "V_gmax0": V_gmax,
    "V_gmax1": V_gmax,
    "V_gmax2": V_gmax,
    "V_gmax3": V_gmax,
    "V_m": 0,
    "V_reset": _HW_parameters["Vreset"],
    "V_stdf": 0,
    "V_thigh": 0,
    "V_tlow": 0,
}


def get_HW_parameters(exp=False, adapt_a=False, adapt_b=False, syn_in_exc=True, syn_in_inh=True):
    """
    returns a dictionairy with default HW parameters.
    exp        - enable exponential term circuit
    adapt_a    - enable adaption parameter 'a'
    adapt_b    - enable adaption parameter 'b'
    syn_in_ex  - enable excitatory synaptic input circuit
    syn_in_inh - enable inhibitory synaptic input circuit
    """

    HW_parameters = dict(_HW_parameters)

    if syn_in_exc:
        HW_parameters["Vsynx"] = 1000    # according to hicann-doc: set to 1V
        HW_parameters["Iintbbx"] = 2000  # according to hicann-doc: set to 2uA
        HW_parameters["gsynx"] = 1000    # TODO: hicann-doc says: usually set to max

    if syn_in_inh:
        HW_parameters["Vsyni"] = 1000    # according to hicann-doc: set to 1V
        HW_parameters["Iintbbi"] = 2000  # according to hicann-doc: set to 2uA
        HW_parameters["gsyni"] = 1000    # TODO: hicann-doc says: usually set to max

    if exp:
        HW_parameters["Ibexp"] = 2000  # nA

    return HW_parameters


def get_global_parameters():
    return dict(_Global_HW_parameters)
