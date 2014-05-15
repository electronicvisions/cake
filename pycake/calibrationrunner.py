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

class CalibrationRunner(object):
    """
    """
    logger = pylogging.get("pycake.calibrationrunner")
    pickle_file_pattern = "runner_{}_{}.p.bz2"
    pickel_measurements_folder = "runner_{}_{}_measurements"

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = pycake.config.Config(None, self.config_file)
        self.experiments = {}
        self.coeffs = {}

        # Initialize calibtic
        path, _ = self.config.get_calibtic_backend()
        wafer, hicann = self.config.get_coordinates()
        self.calibtic = pycake.helpers.calibtic.Calibtic(path, wafer, hicann)

        prefix = self.config.get_filename_prefix()
        self.filename = self.pickle_file_pattern.format(
                prefix, time.strftime('%m%d_%H%M'))
        self.measurements_folder = self.pickel_measurements_folder.format(
                prefix, time.strftime('%m%d_%H%M'))

        self.storage = StorageProcess(compresslevel=9)
        # TODO redman!!

    def run_calibration(self):
        """entry method for regular calibrations
        to resume a calibrations see: continue_calibration"""
        self.clear_calibration() # Clears calibration if this is wanted
        self.logger.INFO("Start calibration")
        config = self.config
        self.experiments.clear()
        self.coeffs.clear()

        for config_name in self.config.get_enabled_calibrations():
            config.set_target(config_name)
            save_traces = config.get_save_traces()
            repetitions = config.get_repetitions()

            self.logger.INFO("Creating analyzers and experiments for "
                    "parameter {}".format(config_name))
            builder = self.get_builder(config_name, config)
            analyzer = builder.get_analyzer(config_name)
            experiments = [
                    self.get_experiment(
                        builder.generate_measurements(), analyzer, save_traces)
                    for _ in range(repetitions)]

            if self.config.get_save_traces():
                for exp_id, exp in enumerate(experiments):
                    for m_id, measurement in enumerate(exp.measurements):
                        measurement.save_traces(
                                self.get_measurement_storage_path(
                                    exp_id, m_id, config_name))

            self.experiments[config_name] = experiments
        self._run_measurements()

    def get_measurement_storage_path(self, experiment_id, measurement_id,
            config_name):
        """ Save measurement i of experiment to a file and clear the traces from
            that measurement.
        """
        folder = os.path.join(
                self.config.get_folder(),
                self.measurements_folder,
                "experiment_{}_{}/".format(config_name, experiment_id))
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
        config = self.config
        for config_name in self.config.get_enabled_calibrations():
            config.set_target(config_name)
            repetitions = config.get_repetitions()
            self.save_state()
            msg = "Running experiment no. {}/{} for {}"
            experiments = self.experiments[config_name]
            for i, ex in enumerate(experiments):
                self.logger.INFO(msg.format(i+1, repetitions, config_name))
                for measured in ex.iter_measurements():
                    if measured:
                        self.save_state()

            self.logger.INFO("Fitting result data for {}".format(config_name))
            calibrator = self.get_calibrator(config_name, experiments)
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

    def get_experiment(self, measurements, analyzer, save_traces):
        """
        """
        return pycake.experiment.BaseExperiment(measurements, analyzer, save_traces)

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
    pickel_measurements_folder = "testrunner_{}_{}_measurements"

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
