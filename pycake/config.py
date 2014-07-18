import imp
import os
from pycake.helpers.units import Voltage, Current, DAC
from pyhalbe.HICANN import neuron_parameter, shared_parameter
import copy

class Config(object):
    def __init__(self, name, parameters):
        self.config_name = name
        if isinstance(parameters, dict):
            self.parameters = copy.deepcopy(parameters)
        else:
            self.parameters = self.read_parameter_file(parameters)

    def copy(self, config_name):
        return Config(config_name, self.parameters)

    def read_parameter_file(self, parameters_file):
        return imp.load_source('parameters', parameters_file).parameters

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
        """ Returns a list of parameters which have their "run" setting to True
        """
        run = []
        for config_name in self.parameters['parameter_order']:
            try:
                if self.parameters["run_{}".format(config_name)]:
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
        return self.get_config_with_default("wafer_cfg", "")

    def get_parameters(self):
        """ Returns the parameter dictionary.
        """
        params = copy.deepcopy(self.parameters["base_parameters"])
        params.update(self.get_config("parameters"))
        return params

    def get_filename_prefix(self):
        try:
            return self.parameters["filename_prefix"]
        except KeyError:
            return ""

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
        path = self.parameters["backend_c"]
        return (path, name)

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
        except KeyError, AttributeError:
            return self.parameters["save_traces"]

    def get_clear(self):
        """ Returns if calibration should be cleared.
        """
        return self.parameters['clear']

    def get_clear_defects(self):
        """ Returns if calibration should be cleared.
        """
        return self.parameters.get('clear_defects',False)

    def get_run_calibration(self):
        return self.parameters["calibrate"]

    def get_run_test(self):
        return self.parameters["measure"]
