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

import numpy
import pandas
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
        self.done = False
        self.config = config
        self.name = config.config_name
        self.storage_folder = None
        self.storage = StorageProcess(compresslevel=9)
        if experiment is None:
            # Store calibtic instance just for debugging
            experiment = self.get_experiment(calibitic)
            experiment.calibtic = calibitic
            experiment.set_read_fg_values(self.config.get_read_fg_values())
        self.experiment = experiment
        self.set_storage_folder(storage_path)
        self.pandas_store = os.path.join(os.path.split(
                                self.storage_folder)[0], "results.h5")
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
        # numeric_index needs less space on disc and can be read out
        # without loading Coordinate (and saving speed is a bit faster)
        self.save_experiment_results(numeric_index=True)
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
        self.write_calibration(calibtic, trafos,
                               self.config.get_target_bigcap(),
                               self.config.get_target_speedup_I_gl(),
                               self.config.get_target_speedup_I_gladapt(),
                               self.config.get_target_speedup_I_radapt())
        self.write_defects(redman, trafos)
        self.save_calibration_results(trafos)
        self.done = True

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

    def write_calibration(self, calibtic, transformations, bigcap,
                          speedup_I_gl, speedup_I_gladapt, speedup_I_radapt):
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
            calibtic.write_calibration(parameter, data, bigcap,
                                       speedup_I_gl, speedup_I_gladapt,
                                       speedup_I_radapt)

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

    @staticmethod
    def check_name(name, keys):
        """
        checks if the string 'name' or name_i for i in N is in
        the list 'keys'

        Args:
            name: [str] string to check for
            keys: [list] list of strings to check in
        Returns:
            name_copy: [str] name appended by '_{}'.format(i+1)
        """
        name_copy = name
        i = 0
        while (name_copy in keys):
            if i > 0:
                name_copy = name_copy.replace(str(i-1), str(i))
            elif i == 0:
                name_copy += '_{}'.format(i)
            i += 1
        return name_copy

    def save_calibration_results(self, transformations):
        """
        extracts calibration data from calibtic object and writes to
        'self.pandas_store'

        Args:
            transformations: [list] of (parameter, dict). dict contains
                             pycalibtic transformations, dict.keys() is
                             NeuronOnHICANN or FGBlockOnHICANN
        """
        column_names = ['neuron', 'shared_block', 'defect', 'degree',
                        'trafo', 'domain_min', 'domain_max']

        for parameter, data in transformations:
            name = parameter if isinstance(parameter, basestring) \
                             else parameter.name
            name += '_calib'
            # Transform FGBlock indexed data to Neuron indexed
            if isinstance(parameter, shared_parameter):
                data = {nrn: data[nrn.toSharedFGBlockOnHICANN()]
                        for nrn
                        in Coordinate.iter_all(Coordinate.NeuronOnHICANN)}
            data_all = []
            for nrn, trafo in data.iteritems():
                nrn_id = nrn.id().value()
                fgb_id = nrn.toSharedFGBlockOnHICANN().id().value()
                if trafo is None:
                    defect = True
                    data_all.append([nrn_id, fgb_id, defect])
                else:
                    defect = False
                    coeffs = numpy.array(trafo.getData(), ndmin=1,
                                         dtype=numpy.float)
                    # TODO: str objects are very inefficiently saved in h5 and
                    # if a DataFrame is loaded and saved in the same store, the
                    # file size grows unexpectedly
                    # https://github.com/pandas-dev/pandas/issues/2132
                    list_to_save = [nrn_id, fgb_id, defect,
                                    len(coeffs) - 1,
                                    type(trafo).__name__,
                                    trafo.getDomainBoundaries().first,
                                    trafo.getDomainBoundaries().second]
                    list_to_save.extend(coeffs)
                    data_all.append(list_to_save)
            max_coeff_len = max(col[3] for col in data_all if not col[2])
            column_names.extend('coeff{}'.format(ii)
                                for ii in range(max_coeff_len+1))
            df = pandas.DataFrame(data_all, columns=column_names)
            df.set_index(['neuron', 'shared_block'], inplace=True)
            df.sortlevel('neuron', inplace=True)
            with pandas.HDFStore(self.pandas_store,
                                 complevel=9, complib='blosc') as store:
                name = self.check_name(name, store.keys())
                store[name] = df

    def save_experiment_results(self, numeric_index=False):
        """
        creates a pandas.DataFrame containing all measurements of the
        experiment of the unit.
        saves the measurement results to "results.h5"
        Excludes columns with datatype object to save space on disk.
        """
        name = self.name
        results = self.experiment.get_all_data(
                numeric_index=numeric_index)
        # drop columns that cannot be directly mapped to c-types
        # this saves a lot of disk space
        droped_columns = results.select_dtypes(include=[object]).keys()
        for key in droped_columns:
            self.logger.WARN(
                "The column '{}' of {} is dropped in results.h5 to "
                "save disk space!".format(key, name))
        results = results.select_dtypes(exclude=[object])

        # patch 'range' values from config bc. they are not recorded for the
        # evaluation
        params = {}
        for ii, param in enumerate(
                            self.config.parameters[self.name +"_range"]):
            params.update({ii: {n.name + '_config': p.value
                                for n,p in param.iteritems()}})
        to_merge = pandas.DataFrame.from_dict(params, orient='index')
        to_merge.index.names = ['step']
        results = pandas.merge(results.reset_index(), to_merge.reset_index(),
                               on=['step'], how='inner')
        results.set_index(['neuron','shared_block', 'step'], inplace=True)

        with pandas.HDFStore(self.pandas_store,
                             complevel=9, complib='blosc') as store:
            name = self.check_name(name, store.keys())
            store[name] = results

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
            time.strftime('%m%d_%H%M%S'),
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

        return pycake.helpers.calibtic.Calibtic(path, wafer, hicann, self.config.get_PLL())

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

    def continue_calibration(self, only_with_name, skip_fits=False):
        """resumes an calibration run

        This method will first complete unfinished measurements.
        Afterwards the calibrator will be run. This can overwrite the results
        loaded from an previous run."""
        self.logger.INFO("Continue calibration")
        self._run_measurements(only_with_name, skip_fits)

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
                # bigcap and speedup settings should not matter for technical parameters
                calibtic.write_calibration(parameter, data, False, "normal", "normal", "normal")

            elif isinstance(parameter, neuron_parameter):

                trafo = pycalibtic.Constant(base_parameters[parameter].toDAC().value)
                data = {neuron: trafo for neuron in neurons}
                # bigcap and speedup settings should not matter for technical parameters
                calibtic.write_calibration(parameter, data, False, "normal", "normal", "normal")

            else:

                raise RuntimeError("parameter {} neither shared nor of type neuron".format(parameter))

    def _run_measurements(self, only_with_name=None, skip_fits=False):
        """execute the measurement loop
        Parameters:
            only_with_name: [list] runs only the calibration with a given name
        """
        for ii, name in enumerate(self.to_run):
            if only_with_name is None or name in only_with_name:
                unit = self.create_or_load_unit(ii)
                if not unit.done or not skip_fits:
                    unit.run()
                    calibtic = self.load_calibtic()
                    redman = self.load_redman()
                    unit.generate_calibration_data(calibtic, redman)
                    unit.save()
                else:
                    self.logger.INFO("Measurements and fits for {} already done. Going on with next one.".format(name))

    def get_unit_folder(self, ii):
        return os.path.join(self.storage_folder, str(ii))

    def load_calibration_unit(self, ii):
        unit  = CalibrationUnit.load(self.get_unit_folder(ii))
        # compatibility for old units that do not have the member 'done'
        if not hasattr(unit, 'done'):
            unit.done = False
        return unit

    def create_or_load_unit(self, ii):
        """Receives a pickle calibration unit or creates a new one"""
        try:
            return self.load_calibration_unit(ii)
        except UnitNotFound:
            name = self.to_run[ii]
            return CalibrationUnit(
                self.config.copy(name), self.get_unit_folder(ii), self.load_calibtic())

    def load_experiment_results(self):
        """
        load the experiment results pandas.DataFrame from "results.h5"
        for the current calibration
        """
        with pandas.HDFStore(os.path.join(self.storage_folder,
                "results.h5")) as store:
            df = store[self.name]
        return df

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
        progress.info("Save calibration runner to {}".format(
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
