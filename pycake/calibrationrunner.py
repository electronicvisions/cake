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
        self.storage = StorageProcess(compresslevel=4)
        # TODO redman!!

    def run_calibration(self):
        """entry method for regular calibrations
        to resume a calibrations see: continue_calibration"""
        self.clear_calibration() # Clears calibration if this is wanted
        self.logger.INFO("Start calibration")
        config = self.config
        self.experiments.clear()
        self.coeffs.clear()

        for parameter in self.config.get_enabled_calibrations():
            config.set_target(parameter)
            save_traces = config.get_save_traces()
            repetitions = config.get_repetitions()

            self.logger.INFO("Creating analyzers and experiments for "
                    "parameter {}".format(parameter.name))
            builder = self.get_builder(parameter, config)
            analyzer = builder.get_analyzer(parameter)
            experiments = [
                    self.get_experiment(
                        builder.generate_measurements(), analyzer, save_traces)
                    for _ in range(repetitions)]
            self.experiments[parameter] = experiments
        self._run_measurements()

    def save_measurement(self, experiment_id, measurement_id, experiment, parameter):
        """ Save measurement i of experiment to a file and clear the traces from
            that measurement.
        """
        param_name = parameter.name
        measurement = experiment.get_measurement(measurement_id)
        top_folder = self.config.get_folder()
        runner_folder = "{}.measurements/".format(self.filename)                    # e.g. runner_0705_1246_measurements/
        experiment_folder = "experiment_{}_{}/".format(param_name, experiment_id) # e.g. experiment_E_l/
        measurement_filename = "measurement_{}.p".format(measurement_id)

        folder = os.path.join(top_folder, runner_folder, experiment_folder)
        pycake.helpers.misc.mkdir_p(folder)

        fullpath = os.path.join(folder, measurement_filename)
        self.logger.INFO("Pickling measurement {} of experiment {}({}) to {}".format(measurement_id, param_name, experiment_id, fullpath))
        cPickle.dump(measurement, open(fullpath, 'wb'), protocol=2)
        self.logger.INFO("Clearing traces of measurement {} of experiment {}({}) from memory.".format(measurement_id, param_name, experiment_id, fullpath))
        measurement.clear_traces()

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
        for parameter in self.config.get_enabled_calibrations():
            config.set_target(parameter)
            repetitions = config.get_repetitions()
            self.save_state()
            msg = "Running experiment no. {}/{} for parameter {}"
            for i, ex in enumerate(self.experiments[parameter]):
                self.logger.INFO(msg.format(i+1, repetitions, parameter.name))
                for measured in ex.iter_measurements():
                    if measured:
                        if config.get_save_traces():
                            self.save_measurement(i, measurement_id, ex, parameter)
                        self.save_state()

            self.logger.INFO("Fitting result data for parameter {}".format(
                parameter.name))
            calibrator = self.get_calibrator(parameter, self.experiments)
            coeffs = calibrator.generate_coeffs()
            self.coeffs[parameter] = coeffs
            self.save_state()

            self.logger.INFO("Writing calibration data for parameter {}".format(
                parameter.name))
            self.write_calibration(parameter, coeffs)


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

    def get_builder(self, parameter, config):
        """ Get the right experiment builder.
        """
        builder_type = getattr(pycake.experimentbuilder, "{}_Experimentbuilder".format(parameter.name))
        return builder_type(config)

    def get_experiment(self, measurements, analyzer, save_traces):
        """
        """
        return pycake.experiment.BaseExperiment(measurements, analyzer, save_traces)

    def get_calibrator(self, parameter, experiments):
        """
        """
        calibrator_type = getattr(pycake.calibrator, "{}_Calibrator".format(parameter.name))
        exes = experiments[parameter]
        return calibrator_type(exes)
        
    def write_calibration(self, parameter, data):
        """
        """
        data = dict((coord, coeff[::-1]) for coord, coeff in data.iteritems())
        self.calibtic.write_calibration(parameter, data)


class TestRunner(CalibrationRunner):
    logger = pylogging.get("pycake.testrunner")
    pickle_file_pattern = "testrunner_{}.p.bz2"

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = pycake.config.Config(None, self.config_file)

        # Initialize calibtic
        path, name = self.config.get_calibtic_backend()
        wafer, hicann = self.config.get_coordinates()
        self.calibtic = pycake.helpers.calibtic.Calibtic(path, wafer, hicann)

        prefix = self.config.get_filename_prefix()
        self.filename = self.pickle_file_pattern.format(
                time.strftime('%m%d_%H%M'))

    def clear_calibration(self):
        self.logger.TRACE("Not clearing calibration since this is test measurement")
        pass

    def write_calibration(self, parameter, coeffs):
        self.logger.INFO("Writing no calibration since this is test measurement")

    def get_builder(self, parameter, config):
        """ Get the right experiment builder.
        """
        builder_type = getattr(pycake.experimentbuilder, "{}_Experimentbuilder".format(parameter.name))
        return builder_type(config, test = True)
