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

    def get_config_with_default(self, config_key, default):
        config_name = self.target_parameter.name
        key = "{}_{}".format(config_name, config_key)
        return self.experiment_parameters.get(key, default)

    def set_config(self, config_key, value):
        """sets a given key for experiment"""
        config_name = self.target_parameter.name
        key = "{}_{}".format(config_name, config_key)
        self.experiment_parameters[key] = value

    def get_step_value(self, value, apply_calibration):
        if self.target_parameter.name[0] in ['E', 'V']:
            return Voltage(value, apply_calibration=apply_calibration)
        else:
            return Current(value, apply_calibration=apply_calibration)

    def init_experiment(self):
        super(BaseCalibration, self).init_experiment()
        if self._calib_backend is None:
            raise TypeError("can not store results without Calibtic backend")
        if self._red_nrns is None:
            raise TypeError("can not store defects without Redman backend")
        self.folder = os.path.join(self.experiment_parameters["folder"], self.folder)
        self.trace_folder = self.get_config_with_default('traces', None)

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
                result = defaultdict(dict)
                step = {target: self.get_step_value(step_value, apply_calibration)}
                for n in self.get_neurons():
                    result[n] = step
                return result
        if isinstance(target, shared_parameter):
            def make(step_value):
                result = defaultdict(dict)
                step = {target: self.get_step_value(step_value, apply_calibration)}
                for b in Coordinate.iter_all(FGBlockOnHICANN):
                    result[b] = step
                return result

        return [make(v) for v in self.get_config("range")]

    def process_results(self, neuron_ids):
        self.process_calibration_results(neuron_ids, self.target_parameter, 1)

    def store_results(self):
        self.store_calibration_results(self.target_parameter)

    def process_calibration_results(self, neurons, parameter, dim):
        """This base class function can be used by child classes as process_results."""
        # containers for final results
        self.results_mean = defaultdict(list)
        self.results_std = defaultdict(list)
        self.results_polynomial = {}

        # self.all_results contains all measurements, need to untangle
        # repetitions and steps
        # This is done for shared and neuron parameters
        repetition = 1
        step_results = defaultdict(list)
        for result in self.all_results:
            # still in the same step, collect repetitions for averaging
            for neuron in neurons:
                step_results[neuron].append(result[neuron])
            repetition += 1
            if repetition > self.repetitions:
                # step is done; average, store and reset
                for neuron in neurons:
                    mean = HWtoDAC(np.mean(step_results[neuron]), parameter)
                    std = HWtoDAC(np.std(step_results[neuron]), parameter)
                    self.results_mean[neuron].append(mean)
                    self.results_std[neuron].append(std)
                repetition = 1
                step_results = defaultdict(list)

        # For shared parameters: mean over one block
        if isinstance(parameter, shared_parameter):
            def iter_v_reset(block):
                neuron_on_quad = Coordinate.NeuronOnQuad(block.x(), block.y())
                for quad in Coordinate.iter_all(Coordinate.QuadOnHICANN):
                    yield Coordinate.NeuronOnHICANN(quad, neuron_on_quad)
                return

            self.results_mean_shared = defaultdict(list)
            self.results_std_shared = defaultdict(list)
            for block in self.get_blocks():
                neurons_on_block = [n for n in iter_v_reset(block)]
                self.results_mean_shared[block] = np.mean([self.results_mean[n_coord] for n_coord in neurons_on_block], axis = 0)
                self.results_std_shared[block] = np.mean([self.results_std[n_coord] for n_coord in neurons_on_block], axis = 0)

        steps = self.get_steps()

        if isinstance(parameter, shared_parameter):
            for block in self.get_blocks():
                self.results_polynomial[block] = self.do_fit(block, parameter, steps, 
                        self.results_mean_shared[block], self.results_std_shared[block], dim)

        else: #if neuron_parameter:
            for neuron in neurons:
                self.results_polynomial[neuron] = self.do_fit(neuron, parameter, steps, 
                        self.results_mean[neuron], self.results_std[neuron], dim)

    def store_calibration_results(self, parameter, isneuron=None):
        """This base class function can be used by child classes as store_results.

           isneuron - if is set to True (False) a neuron (shared) calibration is filled
                      if set to None, the type of calibration is deduced from 'parameter'

        """
        results = self.results_polynomial
        md = pycalibtic.MetaData()
        md.setAuthor("pycake")
        md.setComment("calibration")

        logger = self.logger

        isneuron = isinstance(parameter, neuron_parameter) if isneuron == None else isneuron

        if isneuron:
            collection = self._calib_nc
            CollectionType = pycalibtic.NeuronCalibration 
            coordinates = self.get_neurons()
            redman = self._red_nrns
        else:
            collection = self._calib_bc
            CollectionType = pycalibtic.SharedCalibration
            coordinates = self.get_blocks()
            redman = None


        for coord in coordinates:
            result = results[coord]
            index = coord.id().value()
            if result is None and redman and not redman.has(coord):
                redman.disable(coord)
                # TODO reset/delete calibtic function for this neuron
            else:  # store in calibtic
                if not collection.exists(index):
                    logger.INFO("No existing calibration data for neuron {} found, creating default dataset".format(index))
                    cal = CollectionType()
                    collection.insert(index, cal)
                collection.at(index).reset(parameter, result)

        self.logger.INFO("Storing calibration results")
        self.store_calibration(md)

    def do_fit(self, coord, parameter, steps, mean, std, dim, swap_fit_x_y=False):
        step_values = [step[coord][parameter].toDAC().value for step in steps]
        # TODO think about the weight thing
        weight = 1./(np.array(std) + 1e-8)  # add a tiny value because std may be zero
        coeffs = np.polynomial.polynomial.polyfit(mean        if not swap_fit_x_y else step_values,
                                                  step_values if not swap_fit_x_y else mean,
                                                  dim,
                                                  w=weight)
        coeffs = coeffs[::-1]

        if self.isbroken(coeffs):
            self.logger.WARN("{} with coefficients {} marked as broken.".format(coord, coeffs))
            return None
        else:
            self.logger.INFO("Neuron {} calibrated successfully with coefficients {}".format(coord, coeffs))
            return create_pycalibtic_polynomial(coeffs)


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

    def get_steps(self):
        return self.get_steps_impl(True)

    def process_calibration_results(self, neuron_ids, parameter, linear_fit=False):
        pass

    def store_calibration_results(self, parameter):
        pass

