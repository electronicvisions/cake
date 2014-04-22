import numpy as np
import pylogging
from collections import defaultdict
import pyhalbe
import pycalibtic
from pyhalbe.Coordinate import NeuronOnHICANN, FGBlockOnHICANN
from pycake.helpers.calibtic import init_backend as init_calibtic
from pycake.helpers.redman import init_backend as init_redman
import pyredman as redman
from pycake.helpers.units import Current, Voltage, DAC
from pycake.helpers.trafos import HWtoDAC, DACtoHW, HCtoDAC, DACtoHC, HWtoHC, HCtoHW
from pycake.helpers.WorkerPool import WorkerPool
import pycake.helpers.misc as misc
import pycake.helpers.sthal as sthal
from pycake.measure import Measurement
from scipy.optimize import curve_fit
from pycake.helpers.calibtic import create_pycalibtic_polynomial

import pycake.experimentbuilder
import pycake.experiment
import pycake.analyzer
import pycake.config
import pycake.measure
import pycake.calibrator

# Import everything needed for saving:
import pickle
import time
import os
import bz2
import imp
import copy
import argparse

# shorter names
Coordinate = pyhalbe.Coordinate
Enum = Coordinate.Enum
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

parameter_order =   [   shared_parameter.V_reset,
                        neuron_parameter.E_syni,
                        neuron_parameter.E_synx,
                        neuron_parameter.E_l,
                        neuron_parameter.V_t,
                        neuron_parameter.I_gl,
                        neuron_parameter.V_syntcx,
                        neuron_parameter.V_syntci,
                    ]


class CalibrationRunner(object):
    """
    """
    def __getstate__(self):
        """ Disable stuff from getting pickled that cannot be pickled.
        """
        odict = self.__dict__.copy()
        del odict['logger']
        del odict['backend_c']
        return odict

    def __setstate__(self, dic):
        # Initialize logger and calibtic backend when unpickling
        dic['logger'] = pylogging.get("pycake.calibrationrunner")
        c_path = dic['calibtic_path']
        dic['backend_c'] = init_calibtic(type='xml', path=c_path)
        self.__dict__.update(dic)

    def __init__(self, config_file, test=False):
        self.config_file = config_file
        self.config = pycake.config.Config(None, self.config_file)
        self.calibtic_path, self.calibtic_filename = self.config.get_calibtic_backend()
        self.make_path(self.calibtic_path)
        self.backend_c = init_calibtic(type='xml', path=self.calibtic_path)
        self.logger = pylogging.get("pycake.calibrationrunner")
        self.test = test
        self.filename = "runner{0:02d}{1:02d}.p".format(time.localtime()[3], time.localtime()[4])
        # TODO redman!!

    def run_calibration(self):
        if self.config.get_clear() and not self.test:
            self.clear_calibration()
        self.experiments = {}
        self.coeffs = {}
        if self.test:
            self.logger.INFO("{} - Start test measurements".format(time.asctime()))
        else:
            self.logger.INFO("{} - Start calibration".format(time.asctime()))

        for parameter in self.config.get_enabled_calibrations():
            config = self.config
            config.set_target(parameter)
            save_traces = config.save_traces()
            repetitions = config.get_repetitions()

            key = self.get_key(parameter)
            
            try: # Try loading existing experiments but only if there are enough
                experiments = self.experiments[parameter]
                assert len(experiments) == repetitions
                self.logger.INFO("{} - Loaded existing experiments for parameter {}".format(time.asctime(), parameter.name))
            except KeyError, AssertionError:
                self.logger.INFO("{} - Creating analyzers and experiments for parameter {}".format(time.asctime(), parameter.name))
                analyzer = self.get_analyzer(parameter, config)
                builder = self.get_builder(parameter, config, self.test)
                measurements = builder.generate()
                experiments = [self.get_experiment(measurements, analyzer, save_traces) for r in range(repetitions)]
                self.experiments[parameter] = experiments

            i = 1
            for ex in self.experiments[parameter]:
                self.logger.INFO("{} - Running experiment {}/{} for parameter {}".format(time.asctime(), i, repetitions, parameter.name))
                ex.run_experiment()
                i+=1
                self.save_state()

            if not self.test:
                self.logger.INFO("{} - Fitting result data for parameter {}".format(time.asctime(), parameter.name))
                calibrator = self.get_calibrator(parameter, key, self.experiments)
                coeffs = calibrator.calibrate()
                self.coeffs[parameter] = coeffs

                self.logger.INFO("{} - Writing calibration data for parameter {}".format(time.asctime(), parameter.name))
                self.write_calibration(self.backend_c, parameter, coeffs)

    def save_state(self):
        """ Saves itself to a file in the given path.
        """
        folder = self.config.get_folder()
        fullpath = os.path.join(folder, self.filename)
        self.logger.INFO("{} - Pickling current state to {}.".format(time.asctime(), fullpath))
        pickle.dump(self, open(fullpath, 'wb'))

    def clear_calibration(self):
        """ Clears all calibration data.
        """
        path, name = self.config.get_calibtic_backend()
        fullname = name+".xml"
        fullpath = os.path.join(path, fullname)
        if os.path.isfile(fullpath):
            self.logger.INFO("{} - Clearing calibration data by removing file {}".format(time.asctime(), fullpath))
            os.remove(fullpath)

    def make_path(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def get_builder(self, parameter, config, test):
        """ Get the right experiment builder.
        """
        builder_type = getattr(pycake.experimentbuilder, "{}_Experimentbuilder".format(parameter.name))
        return builder_type(config, test)

    def get_analyzer(self, parameter, config):
        """ Get the appropriate analyzer for a specific parameter.
        """
        if parameter == neuron_parameter.I_gl:
            c_w, c_h = config.get_coordinates()
            return pycake.analyzer.I_gl_Analyzer(c_w, c_h)
        else:
            AnalyzerType = getattr(pycake.analyzer, "{}_Analyzer".format(parameter.name))
            return AnalyzerType()

    def get_experiment(self, measurements, analyzer, save_traces):
        """
        """
        return pycake.experiment.BaseExperiment(measurements, analyzer, save_traces)

    def get_calibrator(self, parameter, key, experiments):
        """
        """
        calibrator_type = getattr(pycake.calibrator, "{}_Calibrator".format(parameter.name))
        exes = experiments[parameter]
        return calibrator_type(parameter, key, exes)
        
    def get_key(self, parameter):
        """ Returns appropriate key for each parameter
        """
        # TODO discuss how this should be done
        if parameter == neuron_parameter.V_t:
            return 'max'
        elif parameter == shared_parameter.V_reset:
            return 'baseline'
        else:
            return 'mean'

    def write_calibration(self, backend, parameter, coeffs):
        """ Writes calibration data
            Coefficients are ordered like this:
            [a, b, c] ==> a*x^0 + b*x^1 + c*x^2 + ...
        """
        hc, nc, bc, md = self.load_calibration(backend)
        name = self.calibtic_filename

        if parameter == shared_parameter.V_reset:
            self.write_V_reset_calibration(backend, coeffs)
            return
        
        self.logger.TRACE("Writing {} calibration to backend.".format(parameter.name))
        for coord, coeff in coeffs.iteritems():
            index = coord.id().value()
            reversed_coeffs = coeff[::-1]
            self.logger.TRACE("Creating polynomial with coefficients {}".format(reversed_coeffs))
            polynomial = create_pycalibtic_polynomial((reversed_coeffs[0], reversed_coeffs[1]))
            if not nc.exists(index):
                cal = pycalibtic.NeuronCalibration()
                nc.insert(index, cal)
            nc.at(index).reset(parameter, polynomial)
            self.logger.TRACE("Resetting parameter {} to polynomial {}".format(parameter.name, polynomial))

        backend.store(name, md, hc)

    def write_V_reset_calibration(self, backend, coeffs):
        """ Write V_reset calibration
            This writes two calibrations at once:
            One for the individual neuron readout shift, one for FGBlock calibration.
        """
        self.logger.TRACE("Writing V_reset calibration")
        hc, nc, bc, md = self.load_calibration(backend)
        name = self.calibtic_filename

        for coord, coeff in coeffs.iteritems():
            index = coord.id().value()
            polynomial = create_pycalibtic_polynomial(coeff[::-1])
            if isinstance(coord, Coordinate.NeuronOnHICANN):
                collection = nc
                CalibrationType = pycalibtic.NeuronCalibration
                param_index = 21
            elif isinstance(coord, Coordinate.FGBlockOnHICANN):
                collection = bc
                CalibrationType = pycalibtic.SharedCalibration
                param_index = shared_parameter.V_reset
            if not collection.exists(index):
                cal = CalibrationType()
                collection.insert(index, cal)
            collection.at(index).reset(param_index, polynomial)

        backend.store(name, md, hc)

    def load_calibration(self, calibtic_backend):
        """ Load existing calibration data from backend.
        """
        hc = pycalibtic.HICANNCollection()
        nc = hc.atNeuronCollection()
        bc = hc.atBlockCollection()
        md = pycalibtic.MetaData()

        name = self.calibtic_filename

        # Delete all standard entries. TODO: fix calibtic to use proper standard entries
        for nid in range(512):
            nc.erase(nid)
        for bid in range(4):
            bc.erase(bid)

        try:
            calibtic_backend.load(name, md, hc)
            # load existing calibration:
            nc = hc.atNeuronCollection()
            bc = hc.atBlockCollection()
        except RuntimeError, e:
            if e.message != "data set not found":
                raise RuntimeError(e)
            else:
                # backend does not exist
                pass

        return (hc, nc, bc, md)
