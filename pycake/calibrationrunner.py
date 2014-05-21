import pylogging

from pycake.helpers.redman import init_backend as init_redman
import pyredman as redman

import pycake.experimentbuilder
import pycake.experiment
import pycake.analyzer
import pycake.config
import pycake.measure
import pycake.calibrator
import pycake.helpers.calibtic
import pycake.helpers.misc
from pycake.helpers.StorageProcess import StorageProcess

# Import everything needed for saving:
import time
import os
import copy
from collections import OrderedDict

class CalibrationRunner(object):
    """
    """
    logger = pylogging.get("pycake.calibrationrunner")
    pickle_file_pattern = "runner_{}.p.bz2"
    pickel_measurements_folder = "runner_{}_measurements"

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = pycake.config.Config(None, self.config_file)
        self.experiments = {}
        self.coeffs = {}
        self.configurations = self.config.get_enabled_calibrations()

        # Initialize calibtic
        path, _ = self.config.get_calibtic_backend()
        wafer, hicann = self.config.get_coordinates()
        self.calibtic = pycake.helpers.calibtic.Calibtic(path, wafer, hicann)

        prefix = self.config.get_filename_prefix()
        name_details = "_".join([s for s in [prefix, time.strftime('%m%d_%H%M')] if s])
        self.filename = self.pickle_file_pattern.format(name_details)
        self.measurements_folder = self.pickel_measurements_folder.format(name_details)

        self.storage = StorageProcess(compresslevel=9)
        # TODO redman!!

    def run_calibration(self):
        """entry method for regular calibrations
        to resume a calibrations see: continue_calibration"""
        self.clear_calibration() # Clears calibration if this is wanted
        self.logger.INFO("Start calibration")
        self.experiments.clear()
        self.coeffs.clear()
        for config_name in self.configurations:
            self.experiments[config_name] = None
        self._run_measurements()

    def get_experiment(self, config_name):
        """Returns the experiment for a givin config step"""
        config = self.config.copy(config_name)

        self.logger.INFO("Creating analyzers and experiments for "
                "parameter {}".format(config_name))
        builder = self.get_builder(config_name, config)
        experiment = builder.get_experiment()

        if self.config.get_save_traces():
            for mid, measurement in enumerate(experiment.measurements):
                measurement.save_traces(self.get_measurement_storage_path(
                                mid, config_name))

        return experiment

    def get_measurement_storage_path(self, measurement_id, config_name):
        """ Save measurement i of experiment to a file and clear the traces from
            that measurement.
        """
        folder = os.path.join(
                self.config.get_folder(),
                self.measurements_folder,
                config_name)
        pycake.helpers.misc.mkdir_p(folder)
        filename = "{}.hdf5".format(measurement_id)
        return os.path.join(folder, filename)

    def clear_calibration(self):
        """ Clears calibration if this is set in the configuration
        """
        if self.config.get_clear():
            self.calibtic.clear_calibration()

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

            self.logger.INFO("Fitting result data for {}".format(config_name))
            calibrator = self.get_calibrator(config_name, experiment)
            coeffs = calibrator.generate_coeffs()
            self.logger.INFO("Writing calibration data for {}".format(
                config_name))
            self.write_calibration(coeffs)
            #self.coeffs[config_name] = coeffs
            #self.save_state()

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
        return calibrator_type(experiments)
        
    def write_calibration(self, coeffs):
        """
        """
        for parameter, data in coeffs:
            data = dict((coord, coeff[::-1]) for coord, coeff in data.iteritems())
            self.calibtic.write_calibration(parameter, data)


class TestRunner(CalibrationRunner):
    logger = pylogging.get("pycake.testrunner")
    pickle_file_pattern = "testrunner_{}.p.bz2"
    pickel_measurements_folder = "testrunner_{}_measurements"

    def clear_calibration(self):
        self.logger.TRACE("Not clearing calibration since this is test measurement")
        pass

    def write_calibration(self, _):
        self.logger.INFO("Writing no calibration since this is test measurement")

    def get_builder(self, config_name, config):
        """ Get the right experiment builder.
        """
        builder_type = getattr(pycake.experimentbuilder,
                "{}_Experimentbuilder".format(config_name))
        return builder_type(config, test = True)
