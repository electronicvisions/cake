"""
cake configuration facilities.


Note for authors: Add default values to DEFAULT_PARAMETERS and don't use get
with a defautl value.
"""

import imp
import os
import copy
import errno
import pylogging

from Coordinate import Enum
from Coordinate import FGBlockOnHICANN
from Coordinate import NeuronOnHICANN
from Coordinate import AnalogOnHICANN

logger = pylogging.get("pycake.config")

DEFAULT_PARAMETERS = {
    "analog": AnalogOnHICANN(0),
    "neurons": [NeuronOnHICANN(Enum(i)) for i in range(512)],
    "blocks":  [FGBlockOnHICANN(Enum(i)) for i in range(4)],

    "save_traces": False,
    "save_raw_traces": False,
    "save_after_each_measurement": True,

    "PLL": 100e6,
    "speedup": "normal",
    "bigcap": True,
    "fg_bias": 0,
    "fg_biasn": 0,
    "hicann_version": 4,

    "folder_prefix": "",
    "folder": "./",
    "backend": "./backend",

    "clear": False,
    "clear_defects": False,
    "calibrate": True,
    "measure": False,

    "sim_denmem": None,
    "sim_denmem_cache": None,
    "sim_denmem_maximum_spikes": None,
    "sim_denmem_mc_seed": None,

    "wafer_cfg": "",
    "dump_file" : None,
    "repetitions": 1,
    "input_spikes": {},

    "hwdb": ""
}


class Config(object):
    """Stores all calibration run configuration."""
    def __init__(self, name, parameters):
        """Load configuration from dictionary or parameter file."""
        self.config_name = name
        if isinstance(parameters, dict):
            parameters = copy.deepcopy(parameters)
            params = copy.deepcopy(DEFAULT_PARAMETERS)
            params.update(parameters)
            self.parameters = params
        else:
            self.parameters = self.read_parameter_file(parameters)

    def copy(self, config_name=None):
        """Create copy of current instance for modification."""
        if config_name is None:
            config_name = self.config_name
        return Config(config_name, self.parameters)

    def read_parameter_file(self, parameters_file):
        to_load = [parameters_file]
        loaded = []
        while to_load:
            filename = os.path.abspath(to_load.pop(0))
            try:
                mod = imp.load_source('parameters{}'.format(len(loaded)),
                                      filename)
            except IOError as err:
                if err.errno == errno.ENOENT:
                    logger.error(str(err))
                    raise RuntimeError("Config file '{}' not found!".format(
                        filename))
                else:
                    raise

            loaded.append((filename, mod.parameters))
            for dep in getattr(mod, "extends", []):
                if not os.path.isabs(dep):
                    dep = os.path.join(os.path.dirname(filename), dep)
                to_load.append(dep)

        parameters = copy.deepcopy(DEFAULT_PARAMETERS)
        for filename, parameter in reversed(loaded):
            logger.info("Loaded parameters from {}".format(filename))
            parameters.update(parameter)
        return parameters

    def get_config(self, config_key):
        """ Returns a given key for experiment
        """
        key = "{}_{}".format(self.config_name, config_key)
        return self.parameters[key]

    def get_config_with_default(self, config_key, default):
        """ Returns a given key for experiment or default value if not found
        """
        try:
            return self.get_config(config_key)
        except KeyError:
            return default

    def set_config(self, config_key, value):
        """ Sets a given key for experiment
        """
        key = "{}_{}".format(self.config_name, config_key)
        self.parameters[key] = value

    def get_target(self):
        """ Returns the target parameter.
        """
        return self.config_name

    def set_target(self, parameter):
        """ Sets the target parameter.
            This affects how most functions read out the parameter file.
        """
        self.config_name = parameter

    def get_enabled_calibrations(self):
        """
        Returns a list of calibrations, which are in 'parameter_order' and do
        not have set run_NAME=False.
        """
        run = []
        for config_name in self.parameters['parameter_order']:
            try:
                if self.parameters.get("run_{}".format(config_name), True):
                    run.append(config_name)
            except KeyError:
                continue
        return run

    def get_neurons(self):
        """ Returns a list of all neurons to be calibrated.
        """
        return self.parameters["neurons"]

    def get_blocks(self):
        """ Returns a list of all blocks to be calibrated.
        """
        return self.parameters["blocks"]

    def get_steps(self):
        """ Returns the steps.
        """
        return self.get_config("range")

    def get_folder(self):
        folder = os.path.expanduser(self.parameters['folder'])
        return folder

    def get_wafer_cfg(self):
        return self.parameters["wafer_cfg"]

    def get_dump_file(self):
        return self.parameters["dump_file"]

    def get_PLL(self):
        return self.parameters["PLL"]

    def get_save_after_each_measurement(self):
        return self.parameters["save_after_each_measurement"]

    def get_parameters(self):
        """ Returns the parameter dictionary.
        """
        params = copy.deepcopy(self.parameters["base_parameters"])
        params.update(self.get_config("parameters"))
        return params

    def get_folder_prefix(self):
        try:
            return self.parameters["folder_prefix"]
        except KeyError:
            return ""

    def get_input_spikes(self):
        """Get configured input spikes.

        Format is dict(GbitLinkOnHICANN: dict(L1Address: list(spikes)))
        """
        if "input_spikes" in self.parameters:
            return self.parameters["input_spikes"]
        else:
            return {}

    def get_step_parameters(self, stepvalue):
        """ Returns parameter dict where the target parameter
            is updated with the given stepvalue.

            Args:
                stepvalue:  mV or nA value without unit
        """
        parameters = self.get_parameters()
        parameters.update(stepvalue)
        return parameters

    def get_calibtic_backend(self):
        """ Returns the calibtic path and filename.
        """
        coord_wafer = self.parameters["coord_wafer"]
        coord_hicann = self.parameters["coord_hicann"]
        wafer_id = coord_wafer.value()
        hicann_id = coord_hicann.id().value()
        name = "w{}-h{}".format(int(wafer_id), int(hicann_id))
        path = self.get_backend_path()
        return (path, name)

    def get_backend_path(self):
        """Return path to the calibtic database"""
        try:
            return self.parameters["backend"]
        except KeyError:
            return self.parameters["backend_c"]

    def get_coordinates(self):
        """ Returns coordinates of wafer and hicann.
        """
        coord_wafer = self.parameters["coord_wafer"]
        coord_hicann = self.parameters["coord_hicann"]
        return (coord_wafer, coord_hicann)

    def get_repetitions(self):
        """ Returns the number of repetitions as an int.
        """
        return self.parameters["repetitions"]

    def get_save_traces(self):
        """ Returns wheter or not traces should be saved.
        """
        try:
            return self.get_config("save_traces")
        except (KeyError, AttributeError):
            return self.parameters["save_traces"]

    def get_save_raw_traces(self):
        """ Returns wheter or not traces should be saved.
        """
        try:
            return self.get_config("save_raw_traces")
        except (KeyError, AttributeError):
            return self.parameters["save_traces"]

    def get_clear(self):
        """ Returns if calibration should be cleared.
        """
        return self.parameters['clear']

    def get_clear_defects(self):
        """ Returns if calibration should be cleared.
        """
        return self.parameters['clear_defects']

    def get_run_calibration(self):
        return self.parameters["calibrate"]

    def get_run_test(self):
        return self.parameters["measure"]

    def get_measure_floating_gates(self):
        return self.parameters["measure_floating_gates"]

    def get_sim_denmem(self):
        return self.parameters["sim_denmem"]

    def get_sim_denmem_cache(self):
        return self.parameters["sim_denmem_cache"]

    def get_hicann_version(self):
        return int(self.parameters["hicann_version"])

    def get_sim_denmem_maximum_spikes(self):
        return self.parameters["sim_denmem_maximum_spikes"]

    def get_sim_denmem_mc_seed(self):
        return self.parameters["sim_denmem_mc_seed"]

    def get_speedup(self):
        return self.parameters['speedup']

    def get_bigcap(self):
        return self.parameters['bigcap']

    def get_fg_bias(self):
        return self.parameters['fg_bias']

    def get_fg_biasn(self):
        return self.parameters['fg_biasn']

    def get_recording_time(self, default):
        return self.parameters.get('recording_time', default)

    def get_no_of_spikes(self, default):
        return self.parameters.get('no_of_spikes', default)

    def get_analog(self):
        return self.parameters['analog']

    def get_hwdb(self):
        return self.parameters['hwdb']
