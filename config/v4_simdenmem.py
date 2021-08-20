"""
Repeat HICANN v4 simulations with the sim denmem software
"""

from pyhalco_common import Enum
from pyhalco_hicann_v2 import FGBlockOnHICANN
from pyhalco_hicann_v2 import NeuronOnHICANN
from pycake.helpers import units

MC = False
if MC:
    MC_SEED = 24011985
    NEURONS = [NeuronOnHICANN(Enum(i)) for i in
               list(range(16)) + list(range(100, 116)) + list(range(300, 316)) + list(range(400, 416))]
else:
    MC_SEED = None
    NEURONS = [NeuronOnHICANN(Enum(i)) for i in
               (0, 1, 128, 129, 256, 257, 384, 385)]

extends = ['v4_params.py']

parameters = {
    "folder_prefix": "simdenmem_mc" if MC else "simdenmem",

    # host and port of your simulator server
    "sim_denmem": "vmimas:8123",  # host and port of your simulator

    # Cache folder, please use with care, changes on server side (e.g. switching
    # HICANN version, or pull) are not detected. Use None to deactivate,
    # folder must exist
    "sim_denmem_cache": "/tmp/koke_sim_denmem_cache",

    # Use this seed for Monte Carlo simulations of neurons, None disables MC
    "sim_denmem_mc_seed" : MC_SEED,

    # Sim denmem minimal current to use
    "sim_denmem_min_current" : units.Ampere(50e-9),

    # HICANN version to use
    "hicann_version" : 4,

    "neurons": NEURONS,
}
