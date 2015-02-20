import pylogging

import pyredman as redman

import pycake.experimentbuilder
import pycake.experiment
import pycake.analyzer
import pycake.config
import pycake.measure
import pycake.calibrator
import pycake.helpers.calibtic
import pycake.helpers.redman
import pycake.helpers.misc
from pycake.helpers.StorageProcess import StorageProcess

# Import everything needed for saving:
import time
import os
import copy
from collections import OrderedDict

from pyhalbe import Coordinate

class CalibrationRunner(object):
    """
    """
    logger = pylogging.get("pycake.calibrationrunner")
    pickle_file_pattern = "runner_{}.p.gz"
    pickel_measurements_folder = "runner_{}_measurements"

    def __init__(self, config):
        self.config = config
        self.experiments = {}
        self.coeffs = {}
        self.configurations = self.config.get_enabled_calibrations()

        # Initialize calibtic
        path, _ = self.config.get_calibtic_backend()
        wafer, hicann = self.config.get_coordinates()
        self.calibtic = pycake.helpers.calibtic.Calibtic(path, wafer, hicann)
        self.redman =   pycake.helpers.redman.Redman(path, Coordinate.HICANNGlobal(hicann,wafer))

        name_details = "_".join([s for s in [
            "w{}".format(wafer.value()),
            "h{}".format(hicann.id().value()),
            self.config.get_filename_prefix(),
            time.strftime('%m%d_%H%M'),
        ] if s != ""])

        self.filename = self.pickle_file_pattern.format(name_details)
        self.measurements_folder = self.pickel_measurements_folder.format(name_details)

        self.storage = StorageProcess(compresslevel=9)

    def run_calibration(self):
        """entry method for regular calibrations
        to resume a calibrations see: continue_calibration"""
        self.clear_calibration() # Clears calibration if this is wanted
        self.clear_defects()
        self.logger.INFO("Start calibration")
        self.experiments.clear()
        self.coeffs.clear()
        for config_name in self.configurations:
            self.experiments[config_name] = None
        self._run_measurements()

    def get_experiment(self, config_name):
        """Returns the experiment for a givin config step"""
        config = self.config.copy(config_name)

        builder = self.get_builder(config_name, config)
        self.logger.INFO("Creating analyzers and experiments for parameter {} "
            "using {}".format(config_name, type(builder).__name__))
        experiment = builder.get_experiment()

        if config.get_save_traces():
            experiment.save_traces(
                self.get_measurement_storage_path(config_name))

        return experiment

    def get_measurement_storage_path(self, config_name):
        """ Save measurement i of experiment to a file and clear the traces from
            that measurement.
        """
        folder = os.path.join(
            self.config.get_folder(),
            self.measurements_folder,
            config_name)
        pycake.helpers.misc.mkdir_p(folder)
        return folder

    def clear_calibration(self):
        """ Clears calibration if this is set in the configuration
        """
        if self.config.get_clear():
            self.calibtic.clear_calibration()

    def clear_defects(self):
        """ Clears defects if this is set in the configuration
        """

        if self.config.get_clear_defects():
            self.redman.clear_defects()

    def continue_calibration(self):
        """resumes an calibration run

        This method will first complete unfinished measurements.
        Afterwards the calibrator will be run. This can overwrite the results
        loaded from an previous run."""

        self.logger.INFO("Continue calibration")
        self._run_measurements()

    def _run_measurements(self):
        """execute the measurement loop"""

        if not self.experiments:
            self.logger.WARN("No experiments configured.")

        for config_name in self.configurations:
            experiment = self.experiments[config_name]
            # Create experiments lazy to apply calibration from previous runs
            # correctly
            if experiment is None:
                experiment = self.get_experiment(config_name)
                self.experiments[config_name] = experiment
                self.save_state()

            self.logger.INFO("Running measurements for {}".format(config_name))
            for measured in experiment.iter_measurements():
                if measured:
                    self.save_state()

            self.generate_calibration_data(config_name, experiment)

    def generate_calibration_data(self, config_name, experiment):
        self.logger.INFO("Fitting result data for {}".format(config_name))
        calibrator = self.get_calibrator(config_name, experiment)
        coeffs = calibrator.generate_coeffs()
        self.logger.INFO("Writing calibration data for {}".format(config_name))
        self.write_calibration(coeffs)
        self.write_defects(coeffs)
        self.coeffs[config_name] = coeffs
        self.save_state()

    def save_state(self):
        """ Saves itself to a file in the given path.
        """
        # TODO zip
        folder = self.config.get_folder()
        pycake.helpers.misc.mkdir_p(folder)
        fullpath = os.path.join(folder, self.filename)
        self.logger.INFO("Pickling current state to {}".format(fullpath))
        self.storage.save_object(fullpath, self)

    def make_path(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def get_builder(self, config_name, config):
        """ Get the right experiment builder.
        """
        builder_type = getattr(pycake.experimentbuilder,
                "{}_Experimentbuilder".format(config_name))
        return builder_type(config)

    def get_calibrator(self, config_name, experiments):
        """
        """
        calibrator_type = getattr(pycake.calibrator,
                "{}_Calibrator".format(config_name))
        return calibrator_type(experiments, self.config.copy(config_name))

    def write_calibration(self, coeffs):
        """
        """
        for parameter, data in coeffs:
            data = dict((coord, coeff[::-1]) for coord, coeff in data.iteritems() if coeff is not None)
            self.calibtic.write_calibration(parameter, data)

    def write_defects(self, coeffs):
        """
        """
        for parameter, data in coeffs:
            defects = [coord for coord, coeff in data.iteritems() if coeff == None]
            self.redman.write_defects(defects)

class TestRunner(CalibrationRunner):
    logger = pylogging.get("pycake.testrunner")
    pickle_file_pattern = "testrunner_{}.p.gz"
    pickel_measurements_folder = "testrunner_{}_measurements"

    def generate_calibration_data(self, config_name, experiment):
        self.logger.INFO("Skipping calibration fit since this is a test measurement.")
        return

    def clear_calibration(self):
        self.logger.TRACE("Not clearing calibration since this is test measurement")
        pass

    def get_builder(self, config_name, config):
        """ Get the right experiment builder.
        """
        builder_type = getattr(pycake.experimentbuilder,
                "{}_Experimentbuilder".format(config_name))
        return builder_type(config, test = True)
