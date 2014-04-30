import imp
from pycake.helpers.units import Voltage, Current, DAC
from pyhalbe.HICANN import neuron_parameter, shared_parameter

class Config(object):
    def __init__(self, target_parameter, parameters_file):
        self.target_parameter = target_parameter
        self.parameters = self.read_parameter_file(parameters_file)

    def read_parameter_file(self, parameters_file):
        return imp.load_source('parameters', parameters_file).parameters

    def get_config(self, config_key):
        """ Returns a given key for experiment
        """
        config_name = self.target_parameter.name
        key = "{}_{}".format(config_name, config_key)
        return self.parameters[key]

    def set_config(self, config_key, value):
        """ Sets a given key for experiment
        """
        config_name = self.target_parameter.name
        key = "{}_{}".format(config_name, config_key)
        self.parameters[key] = value

    def get_target(self):
        """ Returns the target parameter.
        """
        return self.target_parameter

    def set_target(self, parameter):
        """ Sets the target parameter.
            This affects how most functions read out the parameter file.
        """
        self.target_parameter = parameter

    def get_E_syn_dist(self):
        """ Returns the distances of E_syni and E_synx to E_l in mV
        """
        i = self.parameters["E_syni_dist"]
        x = self.parameters["E_synx_dist"]
        return {"E_syni": i,
                "E_synx": x}

    def get_enabled_calibrations(self):
        """ Returns a list of parameters which have their "run" setting to True
        """
        run = []
        for parameter in self.parameters['parameter_order']:
            paramname = parameter.name
            try:
                if self.parameters["run_{}".format(paramname)]:
                    run.append(parameter)
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
        return self.parameters['folder']

    def get_parameters(self):
        """ Returns the parameter dictionary.
        """
        params = self.parameters["base_parameters"]
        params.update(self.get_config("parameters"))
        return params

    def get_filename_prefix(self):
        return self.parameters["filename_prefix"]

    def get_step_parameters(self, stepvalue):
        """ Returns parameter dict that is updated for a step value.

            Args:
                stepvalue:  mV or nA value. How should the step be set?
                            This affects the target parameter
        """
        target_name = self.target_parameter.name
        if target_name == 'I':
            unit = Current
        else:
            unit = Voltage
        parameters = self.get_parameters()
        parameters[self.target_parameter] = unit(stepvalue)
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

    def get_run_calibration(self):
        return self.parameters["calibrate"]

    def get_run_test(self):
        return self.parameters["measure"]
