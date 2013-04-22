_HW_parameters = {
        'EL': 1000, # mV
        'Vt': 1000, # mV
        'Vexp': 536, # mV
        'Vreset': 500, # mV
        'Esynx': 900, # mV
        'Esyni': 900, # mV
        'Vsynx': 0, # mV
        'Vsyni': 0, # mV
        'tausynx': 900, # nA
        'tausyni': 900, # nA
        'expAct': 0, # nA
        'b': 0, # nA
        'a': 0, # nA
        'gL': 400, # nA
        'tauref': 2000, # nA
        'tw': 2000, # nA
        "gsynx" : 0.0, # nA
        "gsyni" : 0.0, # nA
        'Iintbbx': 0, # nA
        'Iintbbi': 0, # nA
        'dT': 750, # nA
        'Ispikeamp': 2000 # nA
        }

V_gmax = 800
_Global_HW_parameters = {
    "I_breset"    : 1023,
    "I_bstim"     : 1023,
    "int_op_bias" : 1023,
    "V_bexp"      : 1023, # only right
    "V_bout"      : 1023, # only left
    "V_br"        : 0,
    "V_bstdf"     : 0,
    "V_ccas"      : 800,
    "V_clra"      : 0,    # only left
    "V_clrc"      : 0,    # only right
    "V_dep"       : 0,
    "V_dllres"    : 400,
    "V_dtc"       : 0,
    "V_fac"       : 0,
    "V_gmax0"     : V_gmax,
    "V_gmax1"     : V_gmax,
    "V_gmax2"     : V_gmax,
    "V_gmax3"     : V_gmax,
    "V_m"         : 0,
    "V_reset"     : _HW_parameters["Vreset"],
    "V_stdf"      : 0,
    "V_thigh"     : 0,
    "V_tlow"      : 0,
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
        HW_parameters['Vsynx'] = 1000   # according to hicann-doc: set to 1 V
        HW_parameters['Vsynx'] = 0   # according to hicann-doc: set to 1 V
        HW_parameters['Iintbbx'] = 2000 # according to hicann-doc: set to 2uA
        HW_parameters['gsynx'] = 1000   # TODO: hicann-doc says: usually set to max

    if syn_in_inh:
        HW_parameters['Vsyni'] = 1000   # according to hicann-doc: set to 1 V
        HW_parameters['Vsyni'] = 0   # according to hicann-doc: set to 1 V
        HW_parameters['Iintbbi'] = 2000 # according to hicann-doc: set to 2uA
        HW_parameters['gsyni'] = 1000   # TODO: hicann-doc says: usually set to max

    if exp:
        HW_parameters["expAct"] = 2000

    return HW_parameters

def get_global_parameters():
    return dict(_Global_HW_parameters)
