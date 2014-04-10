


class Config(object):
    def __init__(self, target_parameter, parameters_file):
        self.target_parameter = target_parameter
        self.parameters = self.read_parameter_file(parameters_file)

    def get_config(self, config_key):
        """returns a given key for experiment"""
        config_name = self.target_parameter.name
        key = "{}_{}".format(config_name, config_key)
        return self.parameters[key]

    def set_config(self, config_key, value):
        """sets a given key for experiment"""
        config_name = self.target_parameter.name
        key = "{}_{}".format(config_name, config_key)
        self.parameters[key] = value

    def get_steps(self):
        return self.get_config("range")

    def get_parameters(self):
        """ First creates a dictionary containing all parameters with halbe defaults.
            Then, it updates all entries specified in the parameterfile.
        """
        params = self.parameters["base_parameters"]
        params.update(self.get_config["parameters"])
        return params

    def get_calibtic_backend(self):
        """ Returns the calibtic path and filename.
        """
        coord_wafer = self.parameters["coord_wafer"]
        coord_hicann = self.parameters["coord_hicann"]
        wafer_id = coord_wafer.value()
        hicann_id = coord_hicann.id().value()
        name = "w{}-h{}".format(int(wafer_id), int(hicann_id))
        return (path, name)

    def get_coordinates(self):
        """ Returns coordinates of wafer and hicann
        """
        coord_wafer = self.parameters["coord_wafer"]
        coord_hicann = self.parameters["coord_hicann"]
        return (coord_wafer, coord_hicann)


    def save_traces(self):
        try:
            return self.get_config("save_traces")
        except KeyError:
            return self.parameters["save_traces"]

