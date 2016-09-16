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
import pycalibtic

# Import everything needed for saving:
import time
import os
import copy
import errno
from collections import OrderedDict
from socket import gethostname

from pyhalbe import Coordinate
from pyhalbe.HICANN import neuron_parameter
from pyhalbe.HICANN import shared_parameter

class UnitNotFound(RuntimeError):
    pass


class InvalidPathToUnit(RuntimeError):
    pass


progress = pylogging.get("progress.cake")

class CalibrationUnit(object):
    logger = pylogging.get("pycake.calibrationunit")

    def __init__(self, config, storage_path, calibitic, experiment=None):
        """
        Arguments:
            experiment: needed for conversion of old experiments
        """
        t_start = time.time()
        self.config = config
        self.name = config.config_name
        self.storage_folder = None
        self.storage = StorageProcess(compresslevel=9)
        if experiment is None:
            # Store calibtic instance just for debugging
            experiment = self.get_experiment(calibitic)
            experiment.calibtic = calibitic
        self.experiment = experiment
        self.set_storage_folder(storage_path)
        self.measure_time = -1.0
        self.setup_time = time.time() - t_start
        self.save()
        progress.info("Create CalibrationUnit {}".format(self.name))
        progress.info("Running on {}".format(gethostname()))
        self.logger.info("Running on {}".format(gethostname()))

    def finished(self):
        if self.experiment:
            return self.experiment.finished()
        else:
            return False

    def run(self):
        """ Run the measurement and use calibrator on the measured data.
            If measurement is already done, it is skipped.
            Returns the generated transformation"""
        progress.info("Running measurements for {}".format(self.name))
        t_start = time.time()
        for measured in self.experiment.iter_measurements():
            if measured:
                if self.config.get_save_after_each_measurement():
                    self.save()

        if not self.config.get_save_after_each_measurement():
            self.save()
        self.measure_time = time.time() - t_start

    def generate_transformations(self):
        """ Get the calibrator and apply it on the finished experiment.

            Returns:
                transformations generated by the calibrator
        """
        progress.debug("Fitting result data for {}".format(self.name))
        calibrator = self.get_calibrator()
        if calibrator.target_parameter is None:
            calibrator.generate_transformations()
            return []
        else:
            return calibrator.generate_transformations()

    def generate_calibration_data(self, calibtic, redman):
        """ Applies calibrator fits to the finished experiment and writes
            the generated calibration data to calibtic backend.

            Args:
                calibtic: calibtic helper object
                redman: redman helper object
        """
        if not self.finished():
            raise RuntimeError("The calibration has not been finished yet")
        trafos = self.generate_transformations()
        progress.debug("Writing calibration data for {}".format(self.name))
        self.write_calibration(calibtic, trafos)
        self.write_defects(redman, trafos)

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
        """ Writes calibration data to calibtic backend
            transformation argument must have the following structure:

            [(parameter1, {coord: transformation, coord: ...}),
             (parameter2, {coord: transformation, .......}), ...]
        """
        for parameter, data in transformations:
            total = len(data)
            good = len([d for d in data.itervalues() if d is not None])
            progress.info(
                "Write calibration data for {}: {} of {} are good.".format(
                    parameter, good, total))
            calibtic.write_calibration(parameter, data)

    def write_defects(self, redman, transformations):
        """
        """
        for parameter, data in transformations:
            defects = [coord for coord, t in data.iteritems() if t is None]
            progress.info(
                "Write defects for {}: {} of {} are bad.".format(
                    parameter, len(defects), len(data)))
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
        progress.debug("Loaded calibration unit from '{}'".format(path))
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
        name_details = "_".join([s for s in [
            self.config.get_folder_prefix(),
            "f{}".format(int(self.config.get_PLL()/1e6)),  # PLL in MHz
            "w{}".format(wafer.value()),
            "h{}".format(hicann.id().value()),
            'bigcap' if config.get_bigcap() else 'smallcap',
            config.get_speedup(),
            time.strftime('%m%d_%H%M'),
        ] if s != ""])

        if storage_path is None:
            self.set_storage_folder(
                os.path.join(self.config.get_folder(), name_details))
        else:
            self.set_storage_folder(storage_path)

        self.save()

    @property
    def redman(self):
        return self.load_redman()

    @property
    def calibtic(self):
        return self.load_calibtic()

    def load_calibtic(self):
        """Load calibtic object"""
        path = self.config.get_backend_path()
        self.logger.INFO("Setting calibtic backend path to: {}".format(path))
        wafer, hicann = self.config.get_coordinates()
        return pycake.helpers.calibtic.Calibtic(
            path, wafer, hicann, self.config.get_PLL(),
            self.config.get_bigcap(), self.config.get_speedup())

    def load_redman(self):
        """Load redman helper"""
        path = self.config.get_backend_path()
        wafer, hicann = self.config.get_coordinates()
        return pycake.helpers.redman.Redman(
            path, Coordinate.HICANNGlobal(hicann,wafer))

    def get(self, **kwargs):
        return [self.load_calibration_unit(ii) for ii in
                self.query_calibrations(**kwargs)]

    def get_single(self, **kwargs):
        pos = self.query_calibrations(**kwargs)
        if len(pos) == 0:
            raise KeyError(kwargs)
        elif len(pos) == 1:
            return self.load_calibration_unit(pos[0])
        else:
            raise RuntimeError("Multiple calibrations found")

    def query_calibrations(self, name=None, pos=None):
        """Return calibrations filtered by the given arguments

        Arguments:
            name [str]: Name of the given calibration unit
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
            calibtic = self.load_calibtic()
            calibtic.clear_calibration()

    def clear_defects(self):
        """ Clears defects if this is set in the configuration
        """

        if self.config.get_clear_defects():
            redman = self.load_redman()
            redman.clear_defects()

    def run_calibration(self):
        """entry method for regular calibrations
        to resume a calibrations see: continue_calibration"""
        self.clear_calibration() # Clears calibration if this is wanted
        self.clear_defects()
        self.logger.INFO("Start calibration")
        self._run_measurements()

    def continue_calibration(self, only_with_name):
        """resumes an calibration run

        This method will first complete unfinished measurements.
        Afterwards the calibrator will be run. This can overwrite the results
        loaded from an previous run."""
        self.logger.INFO("Continue calibration")
        self._run_measurements(only_with_name)

    def finalize(self):
        """to be called after calibration is finished
           writes technical paramaters
        """

        self.logger.INFO("Finalizing")

        calibtic = self.load_calibtic()
        base_parameters = self.config.parameters['base_parameters']
        technical_parameters = self.config.parameters.get('technical_parameters', [])

        neurons = self.config.get_neurons()
        blocks = self.config.get_blocks()

        for parameter in technical_parameters:

            if parameter not in base_parameters:
                raise RuntimeError("technical parameter {} not set".format(parameter))

            if isinstance(parameter, shared_parameter):

                trafo = pycalibtic.Constant(base_parameters[parameter].toDAC().value)
                data = {block: trafo for block in blocks}
                calibtic.write_calibration(parameter, data)

            elif isinstance(parameter, neuron_parameter):

                trafo = pycalibtic.Constant(base_parameters[parameter].toDAC().value)
                data = {neuron: trafo for neuron in neurons}
                calibtic.write_calibration(parameter, data)

            else:

                raise RuntimeError("parameter {} neither shared nor of type neuron".format(parameter))

    def _run_measurements(self, only_with_name=None):
        """execute the measurement loop
        Parameters:
            only_with_name: [list] runs only the calibration with a given name
        """
        for ii, name in enumerate(self.to_run):
            if only_with_name is None or name in only_with_name:
                measurement = self.create_or_load_unit(ii)
                measurement.run()
                calibtic = self.load_calibtic()
                redman = self.load_redman()
                measurement.generate_calibration_data(calibtic, redman)

    def get_unit_folder(self, ii):
        return os.path.join(self.storage_folder, str(ii))

    def load_calibration_unit(self, ii):
        return CalibrationUnit.load(self.get_unit_folder(ii))

    def create_or_load_unit(self, ii):
        """Receives a pickle calibration unit or creates a new one"""
        try:
            return self.load_calibration_unit(ii)
        except UnitNotFound:
            name = self.to_run[ii]
            return CalibrationUnit(
                self.config.copy(name), self.get_unit_folder(ii), self.load_calibtic())

    def set_storage_folder(self, folder):
        """Update the pathes to calibration units results"""
        pycake.helpers.misc.mkdir_p(folder)
        self.storage_folder = folder

    @classmethod
    def load(cls, path, backend_path=None):
        filename = os.path.join(path, cls.filename)
        cls.logger.INFO("Loading '{}'".format(filename))
        data = pycake.helpers.misc.load_pickled_file(filename)
        if not isinstance(data, cls):
            raise RuntimeError("Invalid class loaded!")
        data.set_storage_folder(path)
        if backend_path is not None:
            if not os.path.isdir(backend_path):
                raise RuntimeError("Backend path '{}' not found".format(
                    backend_path))
            data.config.parameters['backend'] = backend_path
        return data

    def save(self):
        storage = StorageProcess(compresslevel=9)
        filename = os.path.join(self.storage_folder, self.filename)
        self.logger.INFO("Save calibration runner '{}'".format(
            self.storage_folder))
        progress.info("Save calibration runnter to {}".format(
            filename))
        storage.save_object(filename, self)

class TestRunner(CalibrationRunner):
    logger = pylogging.get("pycake.testrunner")

    def clear_calibration(self):
        self.logger.TRACE(
            "Not clearing calibration since this is test measurement")

    def _run_measurements(self, only_with_name=None):
        """execute the measurement loop"""
        for ii, name in enumerate(self.to_run):
            if only_with_name is None or name in only_with_name:
                measurement = self.create_or_load_unit(ii)
                measurement.run()
