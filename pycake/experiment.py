"""Framework for experiments on the hardware.

Derive your custom experiment classes with functions for setup,
measurement and processing from BaseExperiment or child classes.
"""

import numpy as np
import pylogging
import pyhalbe
import pycalibtic
from pyhalbe.Coordinate import NeuronOnHICANN, FGBlockOnHICANN
from pycake.helpers.calibtic import init_backend as init_calibtic
from pycake.helpers.redman import init_backend as init_redman
import pyredman as redman
from pycake.helpers.units import Current, Voltage, DAC
import pycake.helpers.misc as misc
import pycake.helpers.sthal as sthal
from pycake.measure import Measurement

# Import everything needed for saving:
import time

# shorter names
Coordinate = pyhalbe.Coordinate
Enum = Coordinate.Enum
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter


class BaseExperiment(object):
    """Base class for running experiments on individual neurons.

    Defines experiment base parameters, steps varying parameters.
    Defines measurement procedure and postprocessing.

    Provides a function to run and process an experiment.
    """
    def __getstate__(self):
        """ Disable stuff from getting pickled that cannot be pickled.
        """
        odict = self.__dict__.copy()
        del odict['logger']
        del odict['progress_logger']
        del odict['_calib_backend']
        del odict['_calib_hc']
        del odict['_calib_nc']
        del odict['_calib_bc']
        del odict['_calib_md']
        return odict

    def __setstate__(self, dic):
        # TODO fix loading of pickled experiment. 
        # Right now, calibtic stuff is not loaded, but only empty variables are set
        dic['_calib_backend'] = None
        dic['_calib_hc'] = None
        dic['_calib_nc'] = None
        dic['_calib_bc'] = None
        dic['_calib_md'] = None
        self.__dict__.update(dic)

    def __init__(self, parameters, target_parameter, neuron_ids=range(512), loglevel=None, recording_time=1e-4):
        self.experiment_parameters = parameters
        self.sthal = sthal.StHALContainer(parameters["coord_wafer"], parameters["coord_hicann"])
        self.sthal.recording_time = recording_time

        self.target_parameter = target_parameter

        self.neurons = [NeuronOnHICANN(Enum(n) if isinstance(n, int) else n) for n in neuron_ids]
        self.blocks = [block for block in Coordinate.iter_all(FGBlockOnHICANN)]

        calibtic_backend = init_calibtic(type = 'xml', path = parameters["backend_c"])
        #redman_backend = init_redman(type = 'xml', path = parameters["backend_r"])

        self._calib_backend = calibtic_backend
        self._calib_hc = None
        self._calib_nc = None
        self._calib_bc = None

        #if redman_backend:
        #    self.init_redman(redman_backend)
        #else:
        #    self._red_wafer = None
        #    self._red_hicann = None
        #    self._red_nrns = None

        if calibtic_backend:
            self.init_calibration()

        self.logger = pylogging.get("pycake.experiment.{}".format(self.target_parameter.name))
        self.progress_logger = pylogging.get("pycake.experiment.{}.progress".format(self.target_parameter.name))

    def get_config(self, config_key):
        """returns a given key for experiment"""
        config_name = self.target_parameter.name
        key = "{}_{}".format(config_name, config_key)
        return self.experiment_parameters[key]

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
        """Hook for child classes. Executed by run_experiment(). These are standard parameters."""
        self.save_results = self.experiment_parameters["save_results"]
        self.save_traces = self.experiment_parameters["save_traces"]
        self.bigcap = True
        self.description = "Basic experiment."  # Change this for all child classes

    #def init_redman(self, backend):
    #    """Initialize defect management for given backend."""
    #    # FIXME default coordinates
    #    coord_hglobal = self.sthal.hicann.index()  # grab HICANNGlobal from StHAL
    #    coord_wafer = coord_hglobal.wafer()
    #    coord_hicann = coord_hglobal.on_wafer()
    #    wafer = redman.Wafer(backend, coord_wafer)
    #    if not wafer.hicanns().has(coord_hicann):
    #        raise ValueError("HICANN {} is marked as defect.".format(int(coord_hicann.id())))
    #    hicann = wafer.find(coord_hicann)
    #    self._red_hicann = hicann
    #    self._red_nrns = hicann.neurons()

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
        try:
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

    def get_parameters(self, apply_calibration=False):
        return self.get_halbe_default_parameters(apply_calibration)

    def get_halbe_default_parameters(self, apply_calibration=False):
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

    def get_calibrated(self, parameters, ncal, coord, param, step_id):
        """ Returns calibrated DAC value from Voltage or Current value
            If DAC value is out of range, it is clipped.

            Args:
                parameters: All parameters of the experiment
                ncal: which calibration should be used?
                coord: neuron coordinate
                param: parameter that should be calibrated
            Returns:
                Calibrated DAC value (if calibration existed)
        """
        value = parameters[param]
        dac_value = value.toDAC().value #implicit range check!
        dac_value_uncalibrated = dac_value
        if ncal and value.apply_calibration:
            try:
                calibration = ncal.at(param)
                dac_value = int(round(calibration.apply(dac_value)))
            except (RuntimeError, IndexError),e:
                if step_id == 0:
                    # Only give this warning in first step
                    self.logger.WARN("{}: Parameter {} not calibrated.".format(coord, param.name))
                pass
            except Exception,e:
                raise e

        if dac_value < 0 or dac_value > 1023:
            if self.target_parameter == neuron_parameter.I_gl: # I_gl handled in another way. Maybe do this for other parameters as well.
                msg = "Calibrated value for {} on Neuron {} has value {} out of range. Value clipped to range."
                self.logger.WARN(msg.format(param.name, coord.id(), dac_value))
                if dac_value < 0:
                    dac_value = 10      # I_gl of 0 gives weird results --> set to 10 DAC
                else:
                    dac_value = 1023
            else:
                msg = "Calibrated value for {} on Neuron {} has value {} out of range. Value clipped to range."
                self.logger.WARN(msg.format(param.name, coord.id(), dac_value))
                if dac_value < 0:
                    dac_value = 0      # I_gl of 0 gives weird results --> set to 10 DAC
                else:
                    dac_value = 1023

        return int(round(dac_value))

    def get_step(self, step_id):
        """ Transforms step as float into something that can be processed by prepare_parameters
        """
        step_value = self.get_steps()[step_id]
        if self.target_parameter.name[0] == 'I':
            step = Current(step_value)
        else:
            step = Voltage(step_value)
        return {self.target_parameter: step}

    def prepare_parameters(self, step, step_id):
        """Prepare parameters before writing them to the hardware.

        This includes converting to DAC values, applying calibration and
        merging of the step specific parameters.

        Calibration can be different for each neuron, which leads to different DAC values
        for each neuron, even though the parameter value is the same.

        Parameters are then written into the sthal container

        Args:
            step: dict, an entry of the list given back by get_steps()
        Returns:
            None
        """

        fgc = pyhalbe.HICANN.FGControl()

        base_parameters = self.get_parameters()

        for neuron in self.neurons:
            neuron_id = neuron.id().value()
            parameters = base_parameters[neuron]
            if isinstance(self.target_parameter, neuron_parameter):
                parameters.update(self.get_step(step_id))
            ncal = None
            #self.logger.TRACE("Neuron {} has: {}".format(neuron.id(), self._red_nrns.has(neuron)))
            try:
                ncal = self._calib_nc.at(neuron_id)
            except (IndexError):
                if step_id == 0:
                    self.logger.WARN("No calibration found for neuron {}.".format(neuron.id()))
                pass

            for name, param in neuron_parameter.names.iteritems():
                if name[0] == '_':
                    continue
                value = self.get_calibrated(parameters, ncal, neuron,
                        param, step_id)
                fgc.setNeuron(neuron, param, value)

        for block in Coordinate.iter_all(FGBlockOnHICANN):
            parameters = base_parameters[block]
            if isinstance(self.target_parameter, shared_parameter):
                parameters.update(self.get_step(step_id))

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
                value = self.get_calibrated(parameters, bcal, block,
                        param, step_id)
                fgc.setShared(block, param, value)

        self.sthal.hicann.floating_gates = fgc

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
        return self.get_config('range')

    def run_experiment(self):
        """Run the experiment and process results."""
        self.init_experiment()

        parameters = self.get_parameters()

        steps = self.get_steps()
        num_steps = len(steps)

        self.measurements = []

        # Save default and step parameters and description to files
        # Also save sthal container to extract standard sthal parameters later:

        self.progress_logger.INFO("{} - Experiment {}".format(time.asctime(), self.description))

        for step_id, step in enumerate(steps):
                self.progress_logger.INFO("{} - Step {}/{}".format(time.asctime(),step_id+1, num_steps))
                self.progress_logger.INFO("{} - Preparing measurement.".format(time.asctime()))
                step_parameters = self.prepare_parameters(step, step_id)
                self.progress_logger.INFO("{} - Setting floating gates and measuring.".format(time.asctime()))
                self.measure()

        if self.sthal._connected:
            self.sthal.disconnect()
        

    def measure(self):
        """ Perform measurements for a single step on one or multiple neurons.
            
            Appends a measurement to the experiment's list of measurements.
        """
        readout_shifts = self.get_readout_shifts(self.neurons)
        measurement = Measurement(self.sthal, self.neurons, readout_shifts)
        measurement.run_measurement()
        
        self.measurements.append(measurement)

    def get_readout_shifts(self, neurons):
        """ Get readout shifts (in V) for a list of neurons.
            If no readout shift is saved, a shift of 0 is returned.

            Args:
                neurons: a list of NeuronOnHICANN coordinates
            Returns:
                shifts: a dictionary {neuron: shift (in V)}
        """
        if not isinstance(neurons, list):
            neurons = [neurons]
        shifts = {}
        for neuron in neurons:
            neuron_id = neuron.id().value()
            try:
                # Since readout shift is a constant, return the value for DAC = 0
                shift = self._calib_nc.at(neuron_id).at(21).apply(0) * 1800./1023. * 1e-3 # Convert to mV
                shifts[neuron] = shift
            except:
                self.logger.WARN("No readout shift calibration for neuron {} found. Using unshifted values.".format(neuron))
                shifts[neuron] = 0
        return shifts

