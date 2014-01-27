"""Framework for experiments on the hardware.

Derive your custom experiment classes with functions for setup,
measurement and processing from BaseExperiment or child classes.
"""

import numpy as np
import pylogging
from collections import defaultdict
import pyhalbe
import pycalibtic
import pyredman as redman
from pycake.helpers.units import Current, Voltage, DAC
from pycake.helpers.trafos import HWtoDAC, DACtoHW, HCtoDAC, DACtoHC, HWtoHC, HCtoHW

# Import everything needed for saving:
import pickle
import time
import os

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
    def __init__(self, neuron_ids, sthal_container, calibtic_backend=None, redman_backend=None, loglevel=None):
        self.sthal = sthal_container

        self.neuron_ids = neuron_ids
        self._repetitions = 1

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

        self.logger = pylogging.get("pycake.experiment")
        if not loglevel is None:
            pylogging.set_loglevel(self.logger, pylogging.LogLevel.INFO)

    def init_experiment(self):
        """Hook for child classes. Executed by run_experiment(). These are standard parameters."""
        self.E_syni_dist = None
        self.E_synx_dist = None
        self.save_results = True
        self.save_traces = False
        self.repetitions = 1
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
        self._red_wafer = wafer
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

    def get_parameters(self):
        """Return neuron parameters for this experiment. Values can be of type Current, Voltage or DAC.

        Returns:
            dict of neuron id -> dict of parameters -> values
        """

        # use halbe default parameters
        coord_neuron = Coordinate.NeuronOnHICANN(Enum(0))
        fgc = pyhalbe.HICANN.FGControl()

        result = {}
        for name, param in neuron_parameter.names.iteritems():
            if name[0] in ("E", "V"):
                result[param] = Voltage(fgc.getNeuron(coord_neuron, param))
            elif name[0] == "I":
                result[param] = Current(fgc.getNeuron(coord_neuron, param))
            else:  # "__last_neuron"
                pass

        return defaultdict(lambda: result)

    def get_shared_parameters(self):
        """Return shared parameters for this experiment. Values can be of type Current, Voltage or DAC.

        Returns:
            dict of block id -> dict of parameters -> values
        """
        shared_params = {}
        # use halbe default parameters
        for coord_id in range(4):
            coord_block = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(coord_id))
            fgc = pyhalbe.HICANN.FGControl()

            result = {}
            for name in pyhalbe.HICANN.shared_parameter.names:
                if name[0] == "V" or name[0] == "E":
                    if ((coord_id%2 == 0) and ((name == "V_clrc") or (name == "V_bexp"))):
                        continue
                    elif ((coord_id%2 == 1) and ((name == "V_clra") or (name == "V_bout"))):
                        continue
                    else:
                        result[pyhalbe.HICANN.shared_parameter.names[name]] = Voltage(fgc.getShared(coord_block, pyhalbe.HICANN.shared_parameter.names[name]))
                elif name[0] == "I" or name[0] == "i":
                    result[pyhalbe.HICANN.shared_parameter.names[name]] = Current(fgc.getShared(coord_block, pyhalbe.HICANN.shared_parameter.names[name]))
                else:  # "__last_neuron"n
                    pass
            shared_params[coord_id] = result

        return shared_params

    def prepare_parameters(self, step_id):
        """Prepare parameters before writing them to the hardware.

        This includes converting to DAC values, applying calibration and
        merging of the step specific parameters.

        Calibration can be different for each neuron, which leads to different DAC values
        for each neuron, even though the parameter value is the same.

        Args:
            parameters: dict of neuron ids -> dict of all neuron parameters
                        and values of type Current, Voltage or DAC
            step_id: index of current step
            neuron_ids: list of neuron ids that should get the parameters

        Returns:
            dict of neuron ids -> dict of neuron parameters -> DAC values (int)

            Example: {0: {neuron_parameter.E_l: 400, ...},
                      1: {neuron_parameter.E_l: 450, ...}}
        """

        steps = self.get_steps()
        shared_steps = self.get_shared_steps()
        neuron_ids = self.get_neurons()
        block_ids = range(4)

        step_parameters = [self.get_parameters()[neuron_id] for neuron_id in self.get_neurons()]
        step_shared_parameters = self.get_shared_parameters()

        # FIXME properly handle broken neurons here?
        # First prepare neuron parameters
        for neuron_id in neuron_ids:
            step_parameters[neuron_id].update(steps[neuron_id][step_id])
            coord_neuron = Coordinate.NeuronOnHICANN(Enum(neuron_id))
            broken = not self._red_nrns.has(coord_neuron)
            if broken and step_id == 0: # Only give this info in the first step. 
                self.logger.WARN("Neuron {} not working. Skipping calibration.".format(neuron_id))
            for param, step_cvalue in step_parameters[neuron_id].iteritems():
                # Handle only neuron parameters in this step. Shared parameters applied afterwards
                calibrated_value = HCtoDAC(step_cvalue.value, param)
                uncalibrated_value = calibrated_value
                if step_cvalue.apply_calibration and not broken:
                    # apply calibration
                    try:
                        ncal = self._calib_nc.at(neuron_id)
                        calibration = ncal.at(param)
                        calibrated_value = int(round(calibration.apply(HCtoDAC(step_cvalue.value, param)))) # Convert to DAC and apply calibration to DAC value
                    except (RuntimeError, IndexError),e:
                        pass
                    except Exception,e:
                        raise e
                if calibrated_value < 0:
                    self.logger.WARN("Calibrated {} for neuron {} too low. Setting to original value".format(param.name, neuron_id))
                    calibrated_value = uncalibrated_value
                if calibrated_value > 1023:
                    self.logger.WARN("Calibrated {} for neuron {} too high. Setting to original value.".format(param.name, neuron_id))
                    calibrated_value = uncalibrated_value
                step_parameters[neuron_id][param] = calibrated_value

            # Set E_syni and E_synx AFTER calibration if set this way
            if self.E_syni_dist and self.E_synx_dist:
                E_l = step_parameters[neuron_id][pyhalbe.HICANN.neuron_parameter.E_l]
                i_dist = HCtoDAC(-self.E_syni_dist, neuron_parameter.E_syni) # Convert mV to DAC
                x_dist = HCtoDAC(self.E_synx_dist, neuron_parameter.E_synx) # Convert mV to DAC
                E_syni = E_l - i_dist
                E_synx = E_l + x_dist
                if not broken: # Try to calibrate if neuron is not labeled as broken
                    try:
                        ncal = self._calib_nc.at(neuron_id)
                        calib_i = ncal.at(neuron_parameter.E_syni)
                        calib_x = ncal.at(neuron_parameter.E_synx)

                        E_syni_new = int(round(calib_i.apply(E_syni)))
                        E_synx_new = int(round(calib_i.apply(E_synx)))
                        if (E_syni_new in range(1024)) and E_synx_new in range(1024):
                            E_syni = E_syni_new
                            E_synx = E_synx_new
                    except (RuntimeError, IndexError),e:
                        self.logger.WARN("E_syn for neuron {} not calibrated! Using uncalibrated value.".format(neuron_id))
                    except OverflowError, e:
                        self.logger.WARN("E_syn calibration for neuron {} invalid. Using original value.".format(neuron_id))
                step_parameters[neuron_id][pyhalbe.HICANN.neuron_parameter.E_syni] = type(step_parameters[neuron_id][pyhalbe.HICANN.neuron_parameter.E_syni])(E_syni)
                step_parameters[neuron_id][pyhalbe.HICANN.neuron_parameter.E_synx] = type(step_parameters[neuron_id][pyhalbe.HICANN.neuron_parameter.E_synx])(E_synx)

        # Now prepare shared parameters
        for block_id in block_ids:
            step_shared_parameters[block_id].update(shared_steps[block_id][step_id])
            coord_block = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(block_id))
            for param in step_shared_parameters[block_id]:
                broken = False
                step_cvalue = step_shared_parameters[block_id][param]
                calibrated_value = HCtoDAC(step_cvalue.value, param)
                uncalibrated_value = calibrated_value
                apply_calibration = step_cvalue.apply_calibration
                if broken:
                    self.logger.WARN("Neuron {} not working. Skipping calibration.".format(neuron_id))
                if apply_calibration and not broken:
                    # apply calibration
                    try:
                        bcal = self._calib_bc.at(block_id)
                        calibration = bcal.at(param)
                        calibrated_value = int(round(calibration.apply(HCtoDAC(step_cvalue.value, param))))
                    except (AttributeError, RuntimeError, IndexError),e:
                        pass
                    except Exception,e:
                        raise e
                # convert to DAC
                if calibrated_value < 0:
                    self.logger.WARN("Calibrated {} for block {} too low. Setting to original value".format(param.name, block_id))
                    calibrated_value = uncalibrated_value
                if calibrated_value > 1023:
                    self.logger.WARN("Calibrated {} for block {} too high. Setting to original value.".format(param.name, block_id))
                    calibrated_value = uncalibrated_value
                step_shared_parameters[block_id][param] = calibrated_value

        return [step_parameters, step_shared_parameters]

    def prepare_measurement(self, step_parameters, step_id, rep_id):
        """Prepare measurement.
        Perform reset, write general hardware settings.
        """
        neuron_parameters = step_parameters[0]
        shared_parameters = step_parameters[1]

        fgc = pyhalbe.HICANN.FGControl()
        # Set neuron parameters for each neuron
        for neuron_id in self.get_neurons():
            coord = Coordinate.NeuronOnHICANN(Enum(neuron_id))
            for parameter in neuron_parameters[neuron_id]:
                value = neuron_parameters[neuron_id][parameter]
                fgc.setNeuron(coord, parameter, value)

        # Set block parameters for each block
        for block_id in range(4):
            coord = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(block_id))
            for parameter in shared_parameters[block_id]:
                value = shared_parameters[block_id][parameter]
                fgc.setShared(coord, parameter, value)

        self.sthal.hicann.floating_gates = fgc

        if self.bigcap is True:
            # TODO add bigcap functionality
            pass

        if self.save_results:
            if not os.path.isdir(os.path.join(self.folder,"floating_gates")):
                os.mkdir(os.path.join(self.folder,"floating_gates"))
            pickle.dump(fgc, open(os.path.join(self.folder,"floating_gates", "step{}rep{}.p".format(step_id,rep_id)), 'wb'))
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
            list of neuron dicts of parameter dicts for each step

            Example: {
                        0: { # first step
                            # neuron 0
                            0: {neuron_parameter.E_l: Voltage(400)},
                            1: {neuron_parameter.E_l: Voltage(400)},
                           },
                        1: { # second step
                            0: neuron_parameter.E_l: Voltage(600)},
                            1: neuron_parameter.E_l: Voltage(650)},
                        },
                            0: {neuron_parameter.E_l: Voltage(800)}
                            1: {neuron_parameter.E_l: Voltage(800)}
                        }
                    }
        """

        # single step using default parameters
        return [defaultdict(lambda: {}) for neuron_id in self.get_neurons()]

    def get_shared_steps(self):
        """Measurement shared steps for sweeping. Individual blocks may have
        different steps, but all blocks must have the same total number
        of steps.

        Returns:
            list of block dicts of parameter dicts for each step

            Example: {
                        0: { # first step
                            # neuron 0
                            0: {pyhalbe.HICANN.shared_parameter.V_reset: Voltage(400)},
                            1: {pyhalbe.HICANN.shared_parameter.V_reset: Voltage(400)},
                           },
                        1: { # second step
                            0: pyhalbe.HICANN.shared_parameter.V_reset: Voltage(600)},
                            1: pyhalbe.HICANN.shared_parameter.V_reset: Voltage(650)},
                        },
                            0: {pyhalbe.HICANN.shared_parameter.V_reset: Voltage(800)}
                            1: {pyhalbe.HICANN.shared_parameter.V_reset: Voltage(800)}
                        }
                    }
        """

        # single step using default parameters
        return [defaultdict(lambda: {}) for block_id in range(4)]



    def get_neurons(self):
        """Which neurons should this experiment run on?

        All neurons will be prepared with the same parameters (except for calibration differences)
        and each neuron will be measured in the measure step.
        """

        return self.neuron_ids  # TODO move this to a property

    def run_experiment(self):
        """Run the experiment and process results."""
        logger = self.logger

        self.init_experiment()
        neuron_ids = self.get_neurons()
        parameters = self.get_parameters()
        shared_parameters = self.get_shared_parameters()

        self.all_results = []
        steps = self.get_steps()
        shared_steps = self.get_shared_steps()
        num_steps = len(steps[neuron_ids[0]])
        num_shared_steps = len(shared_steps[0])

        # Give info about nonexisting calibrations:
        for param in parameters[0]:
            try:
                if isinstance(param, pyhalbe.HICANN.neuron_parameter):
                    ncal = self._calib_nc.at(0).at(param)
            except (RuntimeError, IndexError, AttributeError):
                self.logger.WARN("No calibration found for {}. Using uncalibrated values.".format(param))
        for param in shared_parameters[0]:
            try:
                if isinstance(param, pyhalbe.HICANN.shared_parameter):
                    bcal = self._calib_bc.at(0).at(param)
            except (RuntimeError, IndexError, AttributeError):
                self.logger.WARN("No calibration found for {}. Using uncalibrated values.".format(param))

        # Save default and step parameters and description to files
        # Also save sthal container to extract standard sthal parameters later:

        if self.save_results:
            if not os.path.isdir(self.folder):
                os.mkdir(self.folder)
            pickle.dump(self.sthal.hicann, open(os.path.join(self.folder,'sthalcontainer.p'),"wb"))
            pickle.dump(self.sthal.status(), open(os.path.join(self.folder,'wafer_status.p'),'wb'))
            pickle.dump(self.repetitions, open(os.path.join(self.folder,"repetitions.p"), 'wb'))
            open(os.path.join(self.folder,'description.txt'), 'w').write(self.description)

            # dump neuron parameters and steps:
            paramdump = {nid: parameters[nid] for nid in self.get_neurons()}
            pickle.dump(paramdump, open(os.path.join(self.folder,"parameters.p"),"wb"))
            stepdump = {sid: steps[0][sid] for sid in range(num_steps)}
            pickle.dump(stepdump, open(os.path.join(self.folder,"steps.p"),"wb"))

            # dump shared parameters and steps:
            sharedparamdump = {bid: shared_parameters[bid] for bid in range(4)}
            pickle.dump(sharedparamdump, open(os.path.join(self.folder,"shared_parameters.p"),"wb"))
            sharedstepdump = {sid: shared_steps[0][sid] for sid in range(num_shared_steps)}
            pickle.dump(sharedstepdump, open(os.path.join(self.folder,"shared_steps.p"),"wb"))

        logger.INFO("Experiment {}".format(self.description))
        logger.INFO("Created folders in {}".format(self.folder))

        # First check if step numbers match if both shared and neuron parameters are swept!
        if (num_steps > 0 and num_shared_steps is 0) or (num_shared_steps > 0 and num_steps is 0) or (num_steps == num_shared_steps):
            for step_id in range(max(num_steps, num_shared_steps)):
                step_parameters = self.prepare_parameters(step_id)
                for r in range(self.repetitions):
                    logger.INFO("{} - Step {}/{} repetition {}/{}.".format(time.asctime(),step_id+1, max(num_steps, num_shared_steps), r+1, self.repetitions))
                    logger.INFO("{} - Preparing measurement --> setting floating gates".format(time.asctime()))
                    self.prepare_measurement(step_parameters, step_id, r)
                    logger.INFO("{} - Measuring.".format(time.asctime()))
                    self.measure(neuron_ids, step_id, r)
        else:
            logger.WARN("When sweeping shared AND neuron parameters, both need to have same no. of steps.")

        logger.INFO("Processing results")
        self.process_results(neuron_ids)
        self.store_results()

    def measure(self, neuron_ids, step_id, rep_id):
        """Perform measurements for a single step on one or multiple neurons."""
        results = {}
        for neuron_id in neuron_ids:
            self.sthal.switch_analog_output(neuron_id)
            t, v = self.sthal.read_adc()
            # Save traces in files:
            if self.save_traces:
                folder = os.path.join(self.folder,"traces")
                if not os.path.isdir(os.path.join(self.folder,"traces")):
                    os.mkdir(os.path.join(self.folder,"traces"))
                if not os.path.isdir(os.path.join(self.folder,"traces", "step{}rep{}".format(step_id, rep_id))):
                    os.mkdir(os.path.join(self.folder, "traces", "step{}rep{}".format(step_id, rep_id)))
                pickle.dump([t, v], open(os.path.join(self.folder,"traces", "step{}rep{}".format(step_id, rep_id), "neuron_{}.p".format(neuron_id)), 'wb'))
            results[neuron_id] = self.process_trace(t, v, neuron_id, step_id, rep_id)
        # Now store measurements in a file:
        if self.save_results:
            if not os.path.isdir(os.path.join(self.folder,"results/")):
                os.mkdir(os.path.join(self.folder,"results/"))
            pickle.dump(results, open(os.path.join(self.folder,"results/","step{}_rep{}.p".format(step_id, rep_id)), 'wb'))
        self.all_results.append(results)

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        """Hook class for processing measured traces. Should return one value."""
        return 0

    def process_results(self, neuron_ids):
        """Process measured data."""
        pass  # no processing

    def store_results(self):
        """Hook for storing results in child classes."""
        pass


# TODO, playground 





#class Calibrate_a(BaseCalibration):
#    def get_parameters(self):
#        parameters = super(Calibrate_a, self).get_parameters()
#        for neuron_id in self.get_neurons():
#            parameters[neuron_id].update({
#                neuron_parameter.E_l: Voltage(800),
#                neuron_parameter.I_gl: Current(1000),
#                neuron_parameter.V_t: Voltage(700),
#                neuron_parameter.I_radapt: Current(1000),
#                shared_parameter.V_reset: Voltage(500),
#            })
#        return parameters
#
#    def measure(self, neuron_ids):
#        pass  # TODO
#
#
#class Calibrate_b(BaseCalibration):
#    pass
#
#    def measure(self, neuron_ids):
#        pass  # TODO
#
#
#class Calibrate_tau_w(BaseCalibration):
#    pass
#
#    def measure(self, neuron_ids):
#        pass  # TODO
#
#
#class Calibrate_dT(BaseCalibration):
#    pass
#
#    def measure(self, neuron_ids):
#        pass  # TODO
#
#
#class Calibrate_V_th(BaseCalibration):
#    pass
#
#    def measure(self, neuron_ids):
#        pass  # TODO
