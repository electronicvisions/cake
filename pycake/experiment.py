"""Framework for experiments on the hardware.

Derive your custom experiment classes with functions for setup,
measurement and processing from BaseExperiment or child classes.
"""

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

# Import everything needed for saving:
import pickle
import time
import os
import bz2

import copy

# shorter names
Coordinate = pyhalbe.Coordinate
Enum = Coordinate.Enum
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

class BaseExperiment(object):
    """Base class for running experiments on individual neurons.

    Defines experiment base parameters, steps varying parameters, repetitions.
    Defines measurement procedure and postprocessing.

    Provides a function to run and process an experiment.
    """
    target_parameter = None
    denmem_size = 1

    def __init__(self, neuron_ids, sthal_container, parameters, loglevel=None):
        self.experiment_parameters = parameters
        self.sthal = sthal_container

        # TODO only accept NeuronOnHICANN coordinates?
        self.neurons = [NeuronOnHICANN(Enum(n) if isinstance(n, int) else n) for n in neuron_ids]
        self.blocks = [block for block in Coordinate.iter_all(FGBlockOnHICANN)]
        self._repetitions = 1

        calibtic_backend = init_calibtic(type = 'xml', path = parameters["backend_c"])
        redman_backend = init_redman(type = 'xml', path = parameters["backend_r"])

        self._calib_backend = calibtic_backend
        self._calib_hc = None
        self._calib_nc = None
        self._calib_bc = None

        if redman_backend:
            self.init_redman(redman_backend)
        else:
            self._red_wafer = None
            self._red_hicann = None
            self._red_nrns = None

        if calibtic_backend:
            self.init_calibration()

        if self.target_parameter:
            self.logger = pylogging.get("pycake.experiment.{}".format(self.target_parameter.name))
        else:
            self.logger = pylogging.get("pycake.experiment")
        if not loglevel is None:
            pylogging.set_loglevel(self.logger, pylogging.LogLevel.INFO)

        self.trace_folder = None

    def init_experiment(self):
        """Hook for child classes. Executed by run_experiment(). These are standard parameters."""
        self.E_syni_dist = self.experiment_parameters["E_syni_dist"]
        self.E_synx_dist = self.experiment_parameters["E_synx_dist"]
        self.save_results = self.experiment_parameters["save_results"]
        self.save_traces = self.experiment_parameters["save_traces"]
        self.repetitions = self.experiment_parameters["repetitions"]
        self.bigcap = True
        self.description = "Basic experiment."  # Change this for all child classes
        localtime = time.localtime()
        self.folder = "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)

    def init_redman(self, backend):
        """Initialize defect management for given backend."""
        # FIXME default coordinates
        coord_hglobal = self.sthal.hicann.index()  # grab HICANNGlobal from StHAL
        coord_wafer = coord_hglobal.wafer()
        coord_hicann = coord_hglobal.on_wafer()
        wafer = redman.Wafer(backend, coord_wafer)
        if not wafer.hicanns().has(coord_hicann):
            raise ValueError("HICANN {} is marked as defect.".format(int(coord_hicann.id())))
        hicann = wafer.find(coord_hicann)
        self._red_hicann = hicann
        self._red_nrns = hicann.neurons()

    def get_calibtic_name(self):
        # grab Coordinate.HICANNGlobal from StHAL
        c_hg = self.sthal.hicann.index()

        # collection should be named "w<wafer-id>-h<hicann-id>"
        name = "w{}-h{}".format(int(c_hg.wafer().value()), int(c_hg.on_wafer().id()))
        return name

    def init_calibration(self):
        """Initialize Calibtic backend, load existing calibration data."""

        hc = pycalibtic.HICANNCollection()
        nc = hc.atNeuronCollection()
        bc = hc.atBlockCollection()
        md = pycalibtic.MetaData()

        # Delete all standard entries. TODO: fix calibtic to use proper standard entries
        for nid in range(512):
            nc.erase(nid)
        for bid in range(4):
            bc.erase(bid)

        name = self.get_calibtic_name()
        try:  # TODO replace by 'if backend.exists(name)' in the future, when this function is written
            self._calib_backend.load(name, md, hc)
            # load existing calibration:
            nc = hc.atNeuronCollection()
            bc = hc.atBlockCollection()
        except RuntimeError, e:
            if e.message != "data set not found":
                raise RuntimeError(e)
            else:
                # backend does not exist
                pass

        self._calib_hc = hc
        self._calib_nc = nc
        self._calib_bc = bc
        self._calib_md = md

    def store_calibration(self, metadata=None):
        """Write calibration data to backend"""
        self._red_hicann.commit()
        if not metadata:
            metadata = self._calib_md
        self.logger.INFO("Storing calibration into backend")

        name = self.get_calibtic_name()
        self._calib_backend.store(name, metadata, self._calib_hc)

    def get_parameters(self, apply_calibration=False):
        """Return neuron parameters for this experiment. Values can be of type Current, Voltage or DAC.

        Returns:
            dict of neuron id -> dict of parameters -> values
        """

        # use halbe default parameters
        fgc = pyhalbe.HICANN.FGControl()
        result = {}

        for neuron in Coordinate.iter_all(NeuronOnHICANN):
            values = {}
            for name, param in neuron_parameter.names.iteritems():
                if name[0] is not '_':
                    values[param] = DAC(fgc.getNeuron(neuron, param), apply_calibration)
            result[neuron] = values

        # use halbe default parameters
        for block in Coordinate.iter_all(FGBlockOnHICANN):
            values = {}
            even  = block.id().value()%2 == 0
            for name, param in shared_parameter.names.iteritems():
                if (even and (name in ("V_clrc", "V_bexp"))):
                    continue
                if (not even and (name in ("V_clra", "V_bout"))):
                    continue
                if name[0] is not '_':
                    values[param] = DAC(fgc.getShared(block, param))
                else:  # "__last_neuron"n
                    pass
            result[block] = values

        return result

    def get_calibrated(self, parameters, ncal, coord, param):
        value = parameters[param]
        dac_value = value.toDAC().value #implicit range check!
        dac_value_uncalibrated = dac_value
        if ncal and value.apply_calibration:
            try:
                calibration = ncal.at(param)
                dac_value = int(round(calibration.apply(dac_value)))
            except (RuntimeError, IndexError),e:
                pass
            except Exception,e:
                raise e

        # TODO check with Dominik
        if param == neuron_parameter.E_syni and self.E_syni_dist:
            calibrated_E_l = self.get_calibrated(parameters, ncal, coord, neuron_parameter.E_l)
            dac_value = calibrated_E_l + self.E_syni_dist * 1023/1800.

        if param == neuron_parameter.E_synx and self.E_synx_dist:
            calibrated_E_l = self.get_calibrated(parameters, ncal, coord, neuron_parameter.E_l)
            dac_value = calibrated_E_l + self.E_synx_dist * 1023/1800.

        if dac_value < 0 or dac_value > 1023:
            if self.target_parameter is neuron_parameter.I_gl: # I_gl handled in another way. Maybe do this for other parameters as well.
                msg = "Calibrated value for {} on Neuron {} has value {} out of range. Value clipped to range."
                self.logger.WARN(msg.format(param.name, coord.id(), dac_value))
                if dac_value < 0:
                    dac_value = 10      # I_gl of 0 gives weird results --> set to 10 DAC
                else:
                    dac_value = 1023
            else:
                msg = "Calibrated value for {} on Neuron {} has value {} out of range. Using uncalibrated value."
                self.logger.WARN(msg.format(param.name, coord.id(), dac_value))
                dac_value = dac_value_uncalibrated

        return int(round(dac_value))

    def prepare_parameters(self, step, step_id):
        """Prepare parameters before writing them to the hardware.

        This includes converting to DAC values, applying calibration and
        merging of the step specific parameters.

        Calibration can be different for each neuron, which leads to different DAC values
        for each neuron, even though the parameter value is the same.

        Args:
            step: dict, an entry of the list given back by get_steps()
        Returns:
            None
        """

        fgc = pyhalbe.HICANN.FGControl()

        neurons = self.get_neurons()
        base_parameters = self.get_parameters()

        for neuron in Coordinate.iter_all(NeuronOnHICANN):
            parameters = base_parameters[neuron]
            parameters.update(step[neuron])
            ncal = None
            self.logger.TRACE("Neuron {} has: {}".format(neuron.id(), self._red_nrns.has(neuron)))
            if self._red_nrns.has(neuron):
                self.logger.TRACE("Neuron {} not marked as broken".format(neuron.id()))
                try:
                    ncal = self._calib_nc.at(neuron.id().value())
                    if step_id == 0: # Only show this info in first step
                        self.logger.INFO("Calibration for Neuron {} found.".format(neuron.id()))
                except (IndexError):
                    if step_id == 0:
                        self.logger.WARN("No calibration found for neuron {}".format(neuron.id()))
                    pass
            else:
                self.logger.WARN("Neuron {} marked as not working. Skipping calibration.".format(neuron.id()))

            for name, param in neuron_parameter.names.iteritems():
                if name[0] == '_':
                    continue
                value = self.get_calibrated(parameters, ncal, neuron,
                        param)
                fgc.setNeuron(neuron, param, value)

        for block in Coordinate.iter_all(FGBlockOnHICANN):
            parameters = base_parameters[block]
            parameters.update(step[block])

            # apply calibration
            try:
                bcal = self._calib_bc.at(block.id().value())
            except (IndexError):
                if step_id == 0:
                    self.logger.WARN("No calibration found for block {}".format(block))
                bcal = None

            even  = block.id().value()%2 == 0
            for name, param in shared_parameter.names.iteritems():
                if (even and (name in ("V_clrc", "V_bexp"))):
                    continue
                if (not even and (name in ("V_clra", "V_bout"))):
                    continue
                if name[0] == '_':
                    continue
                value = self.get_calibrated(parameters, ncal, neuron,
                        param)
                fgc.setShared(block, param, value)

        self.sthal.hicann.floating_gates = fgc

    def prepare_measurement(self, step_parameters, step_id, rep_id):
        """Prepare measurement. This is done for each repetition,
        Perform reset, write general hardware settings which were set in prepare_parameters.
        """

        if self.bigcap is True:
            # TODO add bigcap functionality
            pass

        if self.save_results:
            fg_folder = os.path.join(self.folder, 'floating_gates')
            self.pickle(self.sthal.hicann.floating_gates, fg_folder, "step{}rep{}.p".format(step_id, rep_id))

        self.sthal.write_config()

    @property
    def repetitions(self):
        """How many times should each step be repeated?"""
        return self._repetitions

    @repetitions.setter
    def repetitions(self, value):
        self._repetitions = int(value)

    def get_steps(self):
        """Measurement steps for sweeping. Individual neurons may have
        different steps, but all neurons must have the same total number
        of steps.

        Returns:
            A list of steps. Each step defines a dictionary for parameter

            Example: [
                { NeuronOnHICANN(0) : {
                                        neuron_parameter.E_l: Voltage(400),
                                        neuron_parameter.V_t: Voltage(400),
                                      },
                  NeuronOnHICANN(1) : {
                                        neuron_parameter.E_l: Voltage(400),
                                        neuron_parameter.V_t: Voltage(400),
                                      }
                  FGBlockOnHICANN(0) : {
                                        shared_parameter.V_reset: Voltage(400),
                                       }
                  },
                # ...
               ] 
        """
        # One step using default parameters (empty dict does not update the default paramters)
        return [defaultdict(dict)]

    def get_neurons(self):
        """Which neurons should this experiment run on?

        All neurons will be prepared with the same parameters (except for calibration differences)
        and each neuron will be measured in the measure step.
        """
        return self.neurons  # TODO move this to a property

    def get_blocks(self):
        """ Return all blocks on this HICANN.
        """

        return self.blocks  # TODO move this to a property

    def run_experiment(self):
        """Run the experiment and process results."""
        logger = self.logger

        self.init_experiment()
        neurons = self.get_neurons()

        parameters = self.get_parameters()

        self.all_results = []
        steps = self.get_steps()
        num_steps = len(steps)

        # Give info about nonexisting calibrations:
        for name, param in neuron_parameter.names.iteritems():
            if name[0] != '_':
                try:
                    ncal = self._calib_nc.at(0).at(param)
                except (RuntimeError, IndexError, AttributeError):
                    self.logger.WARN("No calibration found for {}. Using uncalibrated values.".format(name))
        for name, param in shared_parameter.names.iteritems():
            if name[0] != '_':
                try:
                    bcal = self._calib_bc.at(0).at(param)
                except (RuntimeError, IndexError, AttributeError):
                    self.logger.WARN("No calibration found for {}. Using uncalibrated values.".format(name))

        # Save default and step parameters and description to files
        # Also save sthal container to extract standard sthal parameters later:
        if self.save_results:
            self.pickle(self.experiment_parameters, self.folder, 'parameterfile.p')
            self.pickle(self.sthal.hicann, self.folder, 'sthalcontainer.p')
            self.pickle(self.repetitions, self.folder, "repetitions.p")
            open(os.path.join(self.folder,'description.txt'), 'w').write(self.description)
            self.pickle(self.target_parameter, self.folder, 'target_parameter.p')
                

            # dump neuron parameters and steps:
            self.pickle(parameters, self.folder, "parameters.p")
            self.pickle(steps, self.folder, "steps.p")

            # This connects implicitly to the wafer
            if not self.trace_folder:
                self.pickle(self.sthal.status(), self.folder, 'wafer_status.p')

        logger.INFO("Experiment {}".format(self.description))
        logger.INFO("Created folders in {}".format(self.folder))
        if self.trace_folder:
            logger.INFO("Reading traces from {}.".format(self.trace_folder))

        for step_id, step in enumerate(steps):
                if not self.trace_folder:
                    step_parameters = self.prepare_parameters(step, step_id)
                for r in range(self.repetitions):
                    logger.INFO("{} - Step {}/{} repetition {}/{}.".format(time.asctime(),step_id+1, num_steps, r+1, self.repetitions))
                    logger.INFO("{} - Preparing measurement --> setting floating gates".format(time.asctime()))
                    if not self.trace_folder:
                        self.prepare_measurement(step_parameters, step_id, r)
                    logger.INFO("{} - Measuring.".format(time.asctime()))
                    self.measure(neurons, step_id, r)

        logger.INFO("Processing results")
        self.process_results(neurons)
        self.store_results()
        if self.sthal._connected:
            self.sthal.disconnect()


    def pickle(self, data, folder, filename):
        if not os.path.isdir(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, filename), "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def pickle_compressed(self, data, folder, filename):
        if os.path.splitext(filename)[1] != '.bz2':
            filename += '.bz2'
        if not os.path.isdir(folder):
            os.makedirs(folder)
        with bz2.BZ2File(os.path.join(folder, filename), "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def get_trace_folder(self, step_id, rep_id):
        return os.path.join(self.folder,"traces", "step{}rep{}".format(step_id, rep_id))

    def save_trace(self, t, v, neuron, step_id, rep_id):
        if self.save_traces:
            folder = self.get_trace_folder(step_id, rep_id)
            filename = "neuron_{}.p".format(neuron.id().value())
            self.pickle_compressed([t, v], folder, filename)

    def save_result(self, result, step_id, rep_id):
        if self.save_results:
            folder = os.path.join(self.folder, "results")
            filename = "step{}_rep{}.p".format(step_id, rep_id)
            self.pickle(result, folder, filename)

    def measure(self, neurons, step_id, rep_id):
        """Perform measurements for a single step on one or multiple neurons."""
        results = {}
        for neuron in neurons:
            if self.trace_folder: # If a trace folder is given, load this trace
                this_step_folder = os.path.join(self.trace_folder, 'step{}rep{}/'.format(step_id, rep_id))
                nid = neuron.id().value()
                fullpath = os.path.join(this_step_folder, 'neuron_{}.p.bz2'.format(nid))
                with bz2.BZ2File(fullpath, 'r') as f:
                    t, v = pickle.load(f)
            else: # Else, measure from chip
                self.sthal.switch_analog_output(neuron)
                t, v = self.sthal.read_adc()
                self.save_trace(t, v, neuron, step_id, rep_id)
            results[neuron] = self.process_trace(t, v, neuron, step_id, rep_id)
        # Now store measurements in a file:
        self.save_result(results, step_id, rep_id)
        self.all_results.append(results)

    def correct_for_readout_shift(self, value, neuron):
        neuron_id = neuron.id().value()
        try:
            shift = self._calib_nc.at(neuron_id).at(21).apply(value * 1023/1800.) * 1800./1023.
            return value - shift
        except:
            logger.WARN("No readout shift calibration for neuron {} found. Using unshifted values.".format(neuron))
            return value

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        """Hook class for processing measured traces. Should return one value."""
        return 0

    def process_results(self, neurons):
        """Process measured data."""
        pass  # no processing

    def store_results(self):
        """Hook for storing results in child classes."""
        pass
