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
import errno
from collections import OrderedDict

from pyhalbe import Coordinate


class UnitNotFound(RuntimeError):
    pass


class InvalidPathToUnit(RuntimeError):
    pass


class CalibrationUnit(object):
    logger = pylogging.get("pycake.calibrationrunner")

    def __init__(self, config, storage_path, calibitic, experiment=None):
        """
        Arguments:
            experiment: needed for conversion of old experiments
        """
        self.config = config
        self.name = config.config_name
        self.result = None
        self.storage_folder = None
        self.storage = StorageProcess(compresslevel=9)
        if experiment is None:
            experiment = self.get_experiment(calibitic)
        self.experiment = experiment
        self.set_storage_folder(storage_path)
        self.save()

    def finished(self):
        if self.experiment:
            return self.experiment.finished() and self.result is not None
        else:
            return False

    def run(self):
        """Run the measurement"""
        self.logger.INFO("Running measurements for {}".format(self.name))
        for measured in self.experiment.iter_measurements():
            if measured:
                self.save()

        self.logger.INFO("Fitting result data for {}".format(self.name))
        calibrator = self.get_calibrator()
        self.result = calibrator.generate_coeffs()
        self.save()

    def generate_calibration_data(self, calibtic, redman):
        if not self.finished():
            raise RuntimeError("The calibration has not been finished yet")
        self.logger.INFO("Writing calibration data for {}".format(self.name))
        self.write_calibration(calibtic, self.result)
        self.write_defects(redman, self.result)
        self.save()

    def get_experiment(self, calibitic):
        """ Get the right experiment builder.
        """
        builder_type = getattr(pycake.experimentbuilder,
                "{}_Experimentbuilder".format(self.name))
        builder = builder_type(self.config, self.config.get_run_test(),
                               calibtic_helper=calibitic)
        self.logger.INFO("Creating analyzers and experiments for parameter {} "
                         "using {}".format(self.name, type(builder).__name__))
        return builder.get_experiment()

    def get_calibrator(self):
        """
        """
        calibrator_type = getattr(pycake.calibrator,
                "{}_Calibrator".format(self.name))
        return calibrator_type(self.experiment, self.config)

    def write_calibration(self, calibtic, transformations):
        """
        """
        def reverse(x):
            if x is not None:
                return x[::-1]
            else:
                return None

        for parameter, data in transformations:
            data = dict((nrn, reverse(t)) for nrn, t in data.iteritems())
            calibtic.write_calibration(parameter, data)

    def write_defects(self, redman, transformations):
        """
        """
        for parameter, data in transformations:
            defects = [coord for coord, t in data.iteritems() if t is None]
            redman.write_defects(defects)

    def set_storage_folder(self, folder):
        """Update the pathes to calibration units results"""
        pycake.helpers.misc.mkdir_p(folder)
        self.storage_folder = folder
        if self.config.get_save_traces():
            self.experiment.save_traces(self.storage_folder)

    @staticmethod
    def filename(storage_folder):
        return os.path.join(storage_folder, "experiment.p.gz")

    def save(self):
        self.logger.INFO(
            "Pickling current state to {}".format(self.storage_folder))
        self.storage.save_object(self.filename(self.storage_folder), self)

    @classmethod
    def load(cls, path):
        if not os.path.exists(cls.filename(path)):
            raise UnitNotFound()
        if not os.path.isdir(path):
            raise InvalidPathToUnit("'{}' is not a path")
        cls.logger.INFO("Loading calibration unit '{}'".format(path))
        data = pycake.helpers.misc.load_pickled_file(cls.filename(path))
        if not isinstance(data, cls):
            raise InvalidPathToUnit("'{}' contained invalid data")
        data.set_storage_folder(path)
        return data


class CalibrationRunner(object):
    logger = pylogging.get("pycake.calibrationrunner")
    filename = "runner.p.gz"

    def __init__(self, config, storage_path=None):
        """
        Arguments:
            storage_path: helper to convert old runners easier
        """

        self.config = config
        self.to_run = self.config.get_enabled_calibrations()

        wafer, hicann = self.config.get_coordinates()
        pll = self.config.get_PLL()
        pll_in_MHz = int(pll/1e6)
        name_details = "_".join([s for s in [
            self.config.get_filename_prefix(),
            "f{}".format(pll_in_MHz),
            "w{}".format(wafer.value()),
            "h{}".format(hicann.id().value()),
            time.strftime('%m%d_%H%M'),
        ] if s != ""])

        if storage_path is None:
            self.set_storage_folder(
                os.path.join(self.config.get_folder(), name_details))
        else:
            self.set_storage_folder(storage_path)

        path, _ = self.config.get_calibtic_backend()
        self.calibtic = pycake.helpers.calibtic.Calibtic(
            path, wafer, hicann, pll)
        self.redman = pycake.helpers.redman.Redman(
            path, Coordinate.HICANNGlobal(hicann,wafer))

        self.save()

    def get(self, **kwargs):
        return [self.load_calibration_step(ii) for ii in
                self.query_calibrations(**kwargs)]

    def get_single(self, **kwargs):
        pos = self.query_calibrations(**kwargs)
        if len(pos) == 0:
            raise KeyError(kwargs)
        elif len(pos) == 1:
            return self.load_calibration_step(pos[0])
        else:
            raise RuntimeError("Multiple calibrations found")

    def query_calibrations(self, name=None, pos=None):
        """Return calibrations filtered by the given arguments

        Arguments:
            name [str]: Name of the given calibration step
            pos [int/slice/iterable]: index of the request calibrations
        """
        if pos is None:
            pos = xrange(len(self.to_run))
        elif isinstance(pos, int):
            pos = [pos]
        elif isinstance(pos, slice):
            pos = xrange(*pos.indices(len(self.to_run)))
        if name:
            pos = [ii for ii in pos if self.to_run[ii] == name]
        return pos

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

    def run_calibration(self):
        """entry method for regular calibrations
        to resume a calibrations see: continue_calibration"""
        self.clear_calibration() # Clears calibration if this is wanted
        self.clear_defects()
        self.logger.INFO("Start calibration")
        self._run_measurements()

    def continue_calibration(self):
        """resumes an calibration run

        This method will first complete unfinished measurements.
        Afterwards the calibrator will be run. This can overwrite the results
        loaded from an previous run."""
        self.logger.INFO("Continue calibration")
        self._run_measurements()

    def _run_measurements(self):
        """execute the measurement loop"""
        for ii, name in enumerate(self.to_run):
            measurement = self.create_or_load_step(ii)
            measurement.run()
            measurement.generate_calibration_data(self.calibtic, self.redman)

    def get_step_folder(self, ii):
        return os.path.join(self.storage_folder, str(ii))

    def load_calibration_step(self, ii):
        return CalibrationUnit.load(self.get_step_folder(ii))

    def create_or_load_step(self, ii):
        """Receives a pickle calibration step or creates a new one"""
        try:
            return self.load_calibration_step(ii)
        except UnitNotFound:
            name = self.to_run[ii]
            return CalibrationUnit(
                self.config.copy(name), self.get_step_folder(ii), self.calibtic)

    def set_storage_folder(self, folder):
        """Update the pathes to calibration steps results"""
        pycake.helpers.misc.mkdir_p(folder)
        self.storage_folder = folder

    @classmethod
    def load(cls, path):
        filename = os.path.join(path, cls.filename)
        cls.logger.INFO("Loading '{}'".format(filename))
        data = pycake.helpers.misc.load_pickled_file(filename)
        if not isinstance(data, cls):
            raise RuntimeError("Invalid class loaded!")
        data.set_storage_folder(path)
        return data

    def save(self):
        storage = StorageProcess(compresslevel=9)
        filename = os.path.join(self.storage_folder, self.filename)
        self.logger.INFO("Save calibration runner '{}'".format(
            self.storage_folder))
        storage.save_object(filename, self)

class TestRunner(CalibrationRunner):
    logger = pylogging.get("pycake.testrunner")

    def clear_calibration(self):
        self.logger.TRACE(
            "Not clearing calibration since this is test measurement")

    def _run_measurements(self):
        """execute the measurement loop"""
        for ii, name in enumerate(self.to_run):
            measurement = self.create_or_load_step(ii)
            measurement.run()
