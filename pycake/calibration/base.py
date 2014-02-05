#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base calibration class. Used by all calibration experiments."""

import numpy as np
import pylogging
from collections import defaultdict
import pyhalbe
import pycalibtic
import pyredman as redman
from pycake.experiment import BaseExperiment
from pycake.helpers.calibtic import create_pycalibtic_polynomial
from pycake.helpers.units import Unit, Current, Voltage, DAC
from pycake.helpers.trafos import HWtoDAC, DACtoHW, HCtoDAC, DACtoHC, HWtoHC, HCtoHW

# Import everything needed for saving:
import pickle
import time
import os

# shorter names
Coordinate = pyhalbe.Coordinate
Enum = Coordinate.Enum
FGBlockOnHICANN = Coordinate.FGBlockOnHICANN
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter


class BaseCalibration(BaseExperiment):
    """Base class for calibration experiments."""

    def get_config(self, config_key):
        """returns a given key for experiment"""
        config_name = self.target_parameter.name
        key = "{}_{}".format(config_name, config_key)
        return self.experiment_parameters[key]

    def set_config(self, config_key, value):
        """returns a given key for experiment"""
        config_name = self.target_parameter.name
        key = "{}_{}".format(config_name, config_key)
        self.experiment_parameters[key] = value

    def get_step_value(self, value, apply_calibration):
        if self.target_parameter.name[0] in ['E', 'V']:
            return {self.target_parameter:
                    Voltage(value, apply_calibration=apply_calibration)}
        else:
            return {self.target_parameter:
                    Current(value, apply_calibration=apply_calibration)}

    def init_experiment(self):
        super(BaseCalibration, self).init_experiment()
        if self._calib_backend is None:
            raise TypeError("can not store results without Calibtic backend")
        if self._red_nrns is None:
            raise TypeError("can not store defects without Redman backend")
        self.folder = os.path.join(self.experiment_parameters["folder"], self.folder)

        self.base_parameters = self.experiment_parameters["base_parameters"]
        self.base_parameters.update(self.get_config("parameters"))
        for param, value in self.base_parameters.iteritems():
            if not isinstance(param, (neuron_parameter, shared_parameter)):
                raise TypeError('Only neuron_parameter or shared_parameter allowed')
            if not isinstance(value, Unit):
                raise TypeError('Values must be given as Unit not as {}'.format(
                    type(value)))

        self.description =  self.get_config("description")

    def get_parameters(self):
        parameters = super(BaseCalibration, self).get_parameters()
        neuron_parameters = dict( (param, v) for param, v in self.base_parameters.iteritems()
                if isinstance(param, neuron_parameter))
        shared_parameters = dict( (param, v) for param, v in self.base_parameters.iteritems()
                if isinstance(param, shared_parameter))

        for neuron in self.get_neurons():
            parameters[neuron].update(neuron_parameters)
        for block in Coordinate.iter_all(FGBlockOnHICANN):
            parameters[block].update(shared_parameters)
        return parameters

    def get_steps(self):
        return self.get_steps_impl(False)

    def get_steps_impl(self, apply_calibration):
        target = self.target_parameter

        if isinstance(target, neuron_parameter):
            def make(step_value):
                step = {target: self.get_step_value(step_value, apply_calibration)}
                return dict((n, step) for n in self.get_neurons())
        if isinstance(target, shared_parameter):
            def make(step_value):
                step = {target: self.get_step_value(step_value, apply_calibration)}
                return dict((b, step) for b in Coordinate.iter_all(FGBlockOnHICANN))

        return [make(v) for v in self.get_config("range")]

    def process_results(self, neuron_ids):
        self.process_calibration_results(neuron_ids, self.target_parameter, linear_fit=True)

    def store_results(self):
        self.store_calibration_results(self.target_parameter)


    def process_calibration_results(self, neuron_ids, parameter, linear_fit=False):
        """This base class function can be used by child classes as process_results."""
        # containers for final results
        results_mean = defaultdict(list)
        results_std = defaultdict(list)
        results_polynomial = {}
        results_broken = []  # will contain neuron_ids which are broken

        # self.all_results contains all measurements, need to untangle
        # repetitions and steps
        # This is done for shared and neuron parameters
        repetition = 1
        step_results = defaultdict(list)
        for result in self.all_results:
            # still in the same step, collect repetitions for averaging
            for neuron_id in neuron_ids:
                step_results[neuron_id].append(result[neuron_id])
            repetition += 1
            if repetition > self.repetitions:
                # step is done; average, store and reset
                for neuron_id in neuron_ids:
                    results_mean[neuron_id].append(HWtoDAC(np.mean(step_results[neuron_id]), parameter))
                    results_std[neuron_id].append(HWtoDAC(np.std(step_results[neuron_id]), parameter))
                repetition = 1
                step_results = defaultdict(list)

        # For shared parameters: mean over one block
        if type(parameter) is shared_parameter:
            results_mean_shared = defaultdict(list)
            results_std_shared = defaultdict(list)
            for block_id in range(4):
                results_mean_shared[block_id] = np.mean([results_mean[n_id] for n_id in range(block_id,block_id+128)], axis = 0)
                results_std_shared[block_id] = np.mean([results_std[n_id] for n_id in range(block_id,block_id+128)], axis = 0)



        if type(parameter) is shared_parameter:
            all_steps = self.get_shared_steps()
            for block_id in range(4):
                results_mean_shared[block_id] = np.array(results_mean_shared[block_id])
                results_std_shared[block_id] = np.array(results_std_shared[block_id])
                steps = [HCtoDAC(step[parameter].value, parameter, rounded = False) for step in all_steps[block_id]]
                if linear_fit:
                    # linear fit
                    m, b = np.linalg.lstsq(zip(results_mean_shared[block_id], [1]*len(results_mean_shared[block_id])), steps)[0]
                    coeffs = [b, m]
                else:
                    # fit polynomial to results
                    weight = 1./(results_std_shared[block_id] + 1e-8)  # add a tiny value because std may be zero
                    # note that np.polynomial.polynomial.polyfit coefficients have
                    # reverse order compared to np.polyfit
                    # to reverse coefficients: rcoeffs = coeffs[::-1]
                    coeffs = np.polynomial.polynomial.polyfit(results_mean_shared[block_id], steps, 2, w=weight)
                    # TODO check coefficients for broken behavior
                results_polynomial[block_id] = create_pycalibtic_polynomial(coeffs)
            self.results_mean_shared = results_mean_shared
            self.results_std_shared = results_std_shared

        else: #if neuron_parameter:
            all_steps = self.get_steps()
            for neuron_id in neuron_ids:
                results_mean[neuron_id] = np.array(results_mean[neuron_id])
                results_std[neuron_id] = np.array(results_std[neuron_id])
                steps = [HCtoDAC(step[parameter].value, parameter) for step in all_steps[neuron_id]]
                if linear_fit:
                    # linear fit
                    m, b = np.linalg.lstsq(zip(results_mean[neuron_id], [1]*len(results_mean[neuron_id])), steps)[0]
                    coeffs = [b, m]
                    # TODO find criteria for broken neurons. Until now, no broken neurons exist
                    if self.isbroken(coeffs):
                        results_broken.append(neuron_id)
                        self.logger.INFO("Neuron {0} marked as broken with coefficients of {1:.2f} and {2:.2f}".format(neuron_id, m, b))
                    #if parameter is neuron_parameter.E_l:
                    #    if not (m > 1.0 and m < 1.5 and b < 0 and b > -500):
                    #        # this neuron is broken
                    #        results_broken.append(neuron_id)
                else:
                    # fit polynomial to results
                    weight = 1./(results_std[neuron_id] + 1e-8)  # add a tiny value because std may be zero

                    # note that np.polynomial.polynomial.polyfit coefficients have
                    # reverse order compared to np.polyfit
                    # to reverse coefficients: rcoeffs = coeffs[::-1]
                    coeffs = np.polynomial.polynomial.polyfit(results_mean[neuron_id], steps, 2, w=weight)
                    # TODO check coefficients for broken behavior
                self.logger.INFO("Neuron {} calibrated successfully with coefficients {}".format(neuron_id, coeffs))
                results_polynomial[neuron_id] = create_pycalibtic_polynomial(coeffs)
            self.results_mean = results_mean
            self.results_std = results_std

        # make final results available
        self.results_polynomial = results_polynomial
        self.results_broken = results_broken

    def store_calibration_results(self, parameter):
        """This base class function can be used by child classes as store_results."""
        results = self.results_polynomial
        md = pycalibtic.MetaData()
        md.setAuthor("pycake")
        md.setComment("calibration")

        logger = self.logger

        if type(parameter) is neuron_parameter:
            isneuron = True
        else:
            isneuron = False

        if isneuron:
            collection = self._calib_nc
        else:
            collection = self._calib_bc

        nrns = self._red_nrns

        for index in results:
            if isneuron:
                coord = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(index))
            else:
                coord = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(index))

            broken = index in self.results_broken
            if broken:
                if nrns.has(coord):  # not disabled
                    nrns.disable(coord)
                    # TODO reset/delete calibtic function for this neuron
            else:  # store in calibtic
                if not collection.exists(index):
                    logger.INFO("No existing calibration data for neuron {} found, creating default dataset".format(index))
                    if isneuron:
                        cal = pycalibtic.NeuronCalibration()
                    else:
                        cal = pycalibtic.SharedCalibration()
                    collection.insert(index, cal)
                collection.at(index).reset(parameter, results[index])

        self.logger.INFO("Storing calibration results")
        self.store_calibration(md)

    def isbroken(self, coefficients):
        """ Specify the function that that tells us if a neuron is broken based on the fit coefficients.

            Should return True or False
        """
        return False


class BaseTest(BaseCalibration):
    """Base class for calibration test experiments."""
    def init_experiment(self):
        super(BaseTest, self).init_experiment()
        self.description = "TEST OF " + self.description

    def get_parameters(self):
        parameters = super(BaseTest, self).get_parameters()
        for neuron_id in self.get_neurons():
            for param, value in parameters[neuron_id].iteritems():
                value.apply_calibration = True
        return parameters

    def get_shared_parameters(self):
        parameters = super(BaseTest, self).get_shared_parameters()
        for block_id in range(4):
            for param, value in parameters[block_id].iteritems():
                value.apply_calibration = True
        return parameters

    def get_steps(self):
        return self.get_steps_impl(self, True)

    def process_calibration_results(self, neuron_ids, parameter, linear_fit=False):
        pass

    def store_calibration_results(self, parameter):
        pass

