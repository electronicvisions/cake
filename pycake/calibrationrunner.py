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

# Import everything needed for saving:
import cPickle
import time
import os
import bz2
import copy

class CalibrationRunner(object):
    """
    """
    logger = pylogging.get("pycake.calibrationrunner")
    pickle_file_pattern = "runner_{}.p"

    def __init__(self, config_file):
        self.config_file = config_file
        self.config = pycake.config.Config(None, self.config_file)

        # Initialize calibtic
        path, name = self.config.get_calibtic_backend()
        wafer, hicann = self.config.get_coordinates()
        self.calibtic = pycake.helpers.calibtic.Calibtic(path, wafer, hicann)

        prefix = self.config.get_filename_prefix()
        self.filename = os.path.join(prefix, self.pickle_file_pattern.format(
                time.strftime('%m%d_%H%M')))
        # TODO redman!!

    def run_calibration(self):
        self.clear_calibration() # Clears calibration if this is wanted
        self.experiments = {}
        self.coeffs = {}
        self.logger.INFO("Start calibration")
        config = self.config

        for parameter in self.config.get_enabled_calibrations():
            config.set_target(parameter)
            save_traces = config.get_save_traces()
            repetitions = config.get_repetitions()

            self.logger.INFO("Creating analyzers and experiments for parameter {}".format(parameter.name))
            builder = self.get_builder(parameter, config)
            analyzer = builder.get_analyzer(parameter)
            experiments = [
                    self.get_experiment(builder.generate_measurements(), analyzer, save_traces)
                    for r in range(repetitions)]
            self.experiments[parameter] = experiments

            for i, ex in enumerate(self.experiments[parameter]):
                self.logger.INFO("Running experiment no. {}/{} for parameter {}".format(i+1, repetitions, parameter.name))
                for measured in ex.iter_measurements():
                    if measured:
                        # TODO Save traces to separate file and clear memory
                        self.save_state()

            self.logger.INFO("Fitting result data for parameter {}".format(parameter.name))
            calibrator = self.get_calibrator(parameter, self.experiments)
            coeffs = calibrator.generate_coeffs()
            self.coeffs[parameter] = coeffs
            self.save_state()

            self.logger.INFO("Writing calibration data for parameter {}".format(parameter.name))
            self.write_calibration(parameter, coeffs)

    def clear_calibration(self):
        """ Clears calibration if this is set in the config
        """
        if self.config.get_clear():
            self.calibtic.clear_calibration()

    def continue_calibration(self):
        self.logger.INFO("Continue calibration")
        config = self.config
        save_traces = config.get_save_traces()
        repetitions = config.get_repetitions()

        for parameter in self.config.get_enabled_calibrations():
            config.set_target(parameter)
            for i, ex in enumerate(self.experiments[parameter]):
                self.logger.INFO("Running experiment no. {}/{} for parameter {}".format(i+1, repetitions, parameter.name))
                for measured in ex.iter_measurements():
                    if measured:
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
        cPickle.dump(self, open(fullpath, 'wb'), protocol=2)

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
        return calibrator_type(parameter, exes)
        
    def write_calibration(self, parameter, data):
        """
        """
        data = dict((coord, coeff[::-1]) for coord, coeff in data.iteritems())
        self.calibtic.write_calibration(parameter, data)


class TestRunner(CalibrationRunner):
    logger = pylogging.get("pycake.testrunner")
    pickle_file_pattern = "testrunner_{}.p"

    def __init__(self, config_file):
        self.logger = pylogging.get("pycake.testrunner")
        self.config_file = config_file
        self.config = pycake.config.Config(None, self.config_file)

        # Initialize calibtic
        path, name = self.config.get_calibtic_backend()
        wafer, hicann = self.config.get_coordinates()
        self.calibtic = pycake.helpers.calibtic.Calibtic(path, wafer, hicann)

        prefix = self.config.get_filename_prefix()
        self.filename = os.path.join(prefix, self.pickle_file_pattern.format(
                time.strftime('%m%d_%H%M')))

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
