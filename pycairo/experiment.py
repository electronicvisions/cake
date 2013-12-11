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
import pycairo.logic.spikes
from pycairo.helpers.calibtic import create_pycalibtic_polynomial
from pycairo.helpers.sthal import StHALContainer, UpdateAnalogOutputConfigurator
from pycairo.helpers.units import Current, Voltage, DAC

# Import everything needed for saving:
import pickle
import time
import os

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
    def __init__(self, neuron_ids, sthal_container, calibtic_backend=None, redman_backend=None, loglevel=pylogging.LogLevel.INFO):
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

        pylogging.reset()
        pylogging.log_to_cout(loglevel)
        self.logger = pylogging.get("pycairo.experiment")

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

        # grab Coordinate.HICANNGlobal from StHAL
        c_hg = self.sthal.hicann.index()

        # collection should be named "w<wafer-id>-h<hicann-id>"
        #name = "w{}-h{}".format(int(c_hg.wafer().id()), int(c_hg.on_wafer().id()))
        name = 'cairo'

        try:  # TODO replace by 'if backend.exists("cairo")' in the future, when this function is written
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
        self._calib_backend.store("cairo", metadata, self._calib_hc)

    def get_parameters(self):
        """Return neuron parameters for this experiment. Values can be of type Current, Voltage or DAC.

        Returns:
            dict of neuron id -> dict of parameters -> values
        """

        # use halbe default parameters
        coord_neuron = Coordinate.NeuronOnHICANN(Enum(0))
        coord_block = Coordinate.FGBlockOnHICANN(Enum(0))
        fgc = pyhalbe.HICANN.FGControl()

        result = {}
        for name, param in neuron_parameter.names:
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

        step_parameters = self.get_parameters()
        step_shared_parameters = self.get_shared_parameters()


        # FIXME properly handle broken neurons here?
        # First prepare neuron parameters
        for neuron_id in neuron_ids:
            step_parameters[neuron_id].update(steps[neuron_id][step_id])
            coord_neuron = Coordinate.NeuronOnHICANN(Enum(neuron_id))
            broken = not self._red_nrns.has(coord_neuron)
            for param in step_parameters[neuron_id]:
                # Handle only neuron parameters in this step. Shared parameters applied afterwards
                step_cvalue = step_parameters[neuron_id][param]
                apply_calibration = step_cvalue.apply_calibration
                if broken:
                    self.logger.WARN("Neuron {} not working. Skipping calibration.".format(neuron_id))
                if apply_calibration and not broken:
                    if not type(step_cvalue) in (Voltage, Current):
                        raise NotImplementedError("can not apply calibration on DAC value")
                    # apply calibration
                    try:
                        ncal = self._calib_nc.at(neuron_id)
                        calibration = ncal.at(param)
                        calibrated_value = calibration.apply(step_cvalue.value)
                        step_cvalue = type(step_cvalue)(calibrated_value)
                    except (RuntimeError, IndexError):
                        pass
                # convert to DAC
                value = step_cvalue.toDAC().value
                step_parameters[neuron_id][param] = value

            # Set E_syni and E_synx AFTER calibration if set this way
            if self.E_syni_dist and self.E_synx_dist:
                E_l = step_parameters[neuron_id][pyhalbe.HICANN.neuron_parameter.E_l]
                i_dist = int(round(self.E_syni_dist * 1023./1800.)) # Convert mV to DAC
                x_dist = int(round(self.E_synx_dist * 1023./1800.)) # Convert mV to DAC
                step_parameters[neuron_id][pyhalbe.HICANN.neuron_parameter.E_syni] = type(step_parameters[neuron_id][pyhalbe.HICANN.neuron_parameter.E_syni])(E_l + i_dist)
                step_parameters[neuron_id][pyhalbe.HICANN.neuron_parameter.E_synx] = type(step_parameters[neuron_id][pyhalbe.HICANN.neuron_parameter.E_synx])(E_l + x_dist)

        # Now prepare shared parameters
        for block_id in block_ids:
            step_shared_parameters[block_id].update(shared_steps[block_id][step_id])
            coord_block = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(block_id))
            for param in step_shared_parameters[block_id]:
                broken = False
                step_cvalue = step_shared_parameters[block_id][param]
                apply_calibration = step_cvalue.apply_calibration
                if broken:
                    self.logger.WARN("Neuron {} not working. Skipping calibration.".format(neuron_id))
                if apply_calibration and not broken:
                    if not type(step_cvalue) in (Voltage, Current):
                        raise NotImplementedError("can not apply calibration on DAC value")
                    # apply calibration
                    try:
                        bcal = self._calib_bc.at(block_id)
                        calibration = bcal.at(param)
                        calibrated_value = calibration.apply(step_cvalue.value)
                        step_cvalue = type(step_cvalue)(calibrated_value)
                    except (AttributeError, RuntimeError, IndexError):
                        pass
                # convert to DAC
                value = step_cvalue.toDAC().value
                step_shared_parameters[block_id][param] = value

        return [step_parameters, step_shared_parameters]

    def prepare_measurement(self, step_parameters, step_id, rep_id):
        """Prepare measurement.
        Perform reset, write general hardware settings.
        """
        neuron_parameters = step_parameters[0]
        shared_parameters = step_parameters[1]

        fgc = pyhalbe.HICANN.FGControl()
        # Set neuron parameters for each neuron
        for neuron_id in neuron_parameters:
            coord = Coordinate.NeuronOnHICANN(Enum(neuron_id))
            for parameter in neuron_parameters[neuron_id]:
                value = neuron_parameters[neuron_id][parameter]
                fgc.setNeuron(coord, parameter, value)

        # Set block parameters for each block
        for block_id in shared_parameters:
            coord = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(block_id))
            for parameter in shared_parameters[block_id]:
                value = shared_parameters[block_id][parameter]
                fgc.setShared(coord, parameter, value)

        self.sthal.hicann.floating_gates = fgc

        if self.bigcap is True:
            # TODO add bigcap functionality
            pass
        
        if self.save_results:
            if not os.path.isdir(os.path.join(self.folder,"floating_gates/")):
                os.mkdir(os.path.join(self.folder,"floating_gates/"))
            pickle.dump(fgc, open(os.path.join(self.folder,"floating_gates/","step{}rep{}.p".format(step_id,rep_id)), 'wb'))
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
        for param in steps[0]:
            try:
                if type(param) is pyhalbe.HICANN.neuron_parameter:
                    ncal = self._calib_nc.at(0).at(param)
            except (RuntimeError, IndexError, AttributeError):
                self.logger.WARN("No calibration found for {}. Using uncalibrated values.".format(param))
        for param in steps[0]:
            try:
                if type(param) is pyhalbe.HICANN.shared_parameter:
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
            #paramdump = {nid:{parameters[nid].keys()[pid].name: parameters[nid].values()[pid] for pid in parameters[0].keys()} for nid in self.get_neurons()}
            paramdump = {nid: parameters[nid] for nid in self.get_neurons()}
            pickle.dump(paramdump, open(os.path.join(self.folder,"parameters.p"),"wb"))
            #stepdump = [{pid.name: steps[0][sid][pid] for pid in steps[0][sid].keys()} for sid in range(num_steps)] 
            stepdump = {sid: steps[0][sid] for sid in range(num_steps)}
            pickle.dump(stepdump, open(os.path.join(self.folder,"steps.p"),"wb"))

            # dump shared parameters and steps:
            sharedparamdump = {bid: shared_parameters[bid] for bid in range(4)}
            pickle.dump(sharedparamdump, open(os.path.join(self.folder,"shared_parameters.p"),"wb"))
            sharedstepdump = {sid: shared_steps[0][sid] for sid in range(num_shared_steps)}
            pickle.dump(sharedstepdump, open(os.path.join(self.folder,"shared_steps.p"),"wb"))
        
        logger.INFO("Experiment: {}".format(self.description))
        logger.INFO("Finished initializing pickle folders. Starting with measurements.")
        # First check if step numbers match if both shared and neuron parameters are swept!
        if (num_steps > 0 and num_shared_steps is 0) or (num_shared_steps > 0 and num_steps is 0) or (num_steps == num_shared_steps):
            for step_id in range(max(num_steps, num_shared_steps)):
                step_parameters = self.prepare_parameters(step_id)
                for r in range(self.repetitions):
                    pylogging.set_loglevel(self.logger, pylogging.LogLevel.INFO)
                    logger.INFO("Step {} repetition {}.".format(step_id, r))
                    logger.INFO("Preparing measurement --> setting floating gates")
                    pylogging.set_loglevel(self.logger, pylogging.LogLevel.ERROR) # Disable FGBlock messages
                    self.prepare_measurement(step_parameters, step_id, r)
                    pylogging.set_loglevel(self.logger, pylogging.LogLevel.INFO)
                    logger.INFO("Measuring.")
                    pylogging.set_loglevel(self.logger, pylogging.LogLevel.ERROR) # Disable FGBlock messages
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
            self.sthal.adc.record()
            v = self.sthal.adc.trace()
            t = self.sthal.adc.getTimestamps()
            # Save traces in files:
            if self.save_traces:
                if not os.path.isdir(os.path.join(self.folder,"traces/")):
                    os.mkdir(os.path.join(self.folder,"traces/"))
                if not os.path.isdir(os.path.join(self.folder,"traces/","step{}rep{}/".format(step_id, rep_id))):
                    os.mkdir(os.path.join(self.folder,"traces/","step{}rep{}/".format(step_id, rep_id)))
                pickle.dump([t, v], open(os.path.join(self.folder,"traces/","step{}rep{}/".format(step_id, rep_id),"neuron_{}.p".format(neuron_id)), 'wb'))

            results[neuron_id] = self.process_trace(t, v)
        # Now store measurements in a file:
        if self.save_results:
            if not os.path.isdir(os.path.join(self.folder,"results/")):
                os.mkdir(os.path.join(self.folder,"results/"))
            pickle.dump(results, open(os.path.join(self.folder,"results/","step{}_rep{}.p".format(step_id, rep_id)), 'wb'))
        self.all_results.append(results)

    def process_trace(self, t, v):
        """Hook class for processing measured traces. Should return one value."""
        return 0

    def process_results(self, neuron_ids):
        """Process measured data."""
        pass  # no processing

    def store_results(self):
        """Hook for storing results in child classes."""
        pass


class BaseCalibration(BaseExperiment):
    """Base class for calibration experiments."""
    def get_parameters(self):
        parameters = {pyhalbe.HICANN.neuron_parameter.E_l: Voltage(900),
                      pyhalbe.HICANN.neuron_parameter.E_syni: Voltage(800),     # synapse
                      pyhalbe.HICANN.neuron_parameter.E_synx: Voltage(1000),    # synapse
                      pyhalbe.HICANN.neuron_parameter.I_bexp: Current(0),       # exponential term set to 0
                      pyhalbe.HICANN.neuron_parameter.I_convi: Current(2500),   # bias current for synaptic input
                      pyhalbe.HICANN.neuron_parameter.I_convx: Current(2500),   # bias current for synaptic input 
                      pyhalbe.HICANN.neuron_parameter.I_fire: Current(0),       # adaptation term b
                      pyhalbe.HICANN.neuron_parameter.I_gladapt: Current(0),    # adaptation term
                      pyhalbe.HICANN.neuron_parameter.I_gl: Current(1000),      # leakage conductance
                      pyhalbe.HICANN.neuron_parameter.I_intbbi: Current(2000),  # integrator bias in synapse
                      pyhalbe.HICANN.neuron_parameter.I_intbbx: Current(2000),  # integrator bias in synapse
                      pyhalbe.HICANN.neuron_parameter.I_pl: Current(2000),      # tau_refrac
                      pyhalbe.HICANN.neuron_parameter.I_radapt: Current(2000),  # 
                      pyhalbe.HICANN.neuron_parameter.I_rexp: Current(750),     # something about the strength of exp term
                      pyhalbe.HICANN.neuron_parameter.I_spikeamp: Current(2000),#
                      pyhalbe.HICANN.neuron_parameter.V_exp: Voltage(536),      # exponential term
                      pyhalbe.HICANN.neuron_parameter.V_syni: Voltage(1000),    # inhibitory synaptic reversal potential
                      pyhalbe.HICANN.neuron_parameter.V_syntci: Voltage(900),   # inhibitory synapse time constant
                      pyhalbe.HICANN.neuron_parameter.V_syntcx: Voltage(900),   # excitatory synapse time constant
                      pyhalbe.HICANN.neuron_parameter.V_synx: Voltage(1000),    # excitatory synaptic reversal potential 
                      pyhalbe.HICANN.neuron_parameter.V_t: Voltage(1000),       # recommended threshold, maximum is 1100
                      }
        return defaultdict(lambda: dict(parameters))

    def get_shared_parameters(self):
        parameters = super(BaseCalibration, self).get_shared_parameters()
        
        for bid in range(4):
            parameters[bid][pyhalbe.HICANN.shared_parameter.V_reset] = Voltage(500)

        return parameters


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
                    results_mean[neuron_id].append(np.mean(step_results[neuron_id]))
                    results_std[neuron_id].append(np.std(step_results[neuron_id]))
                repetition = 1
                step_results = defaultdict(list)

        # For shared parameters: mean over one block
        if type(parameter) is pyhalbe.HICANN.shared_parameter:
            results_mean_shared = defaultdict(list)
            results_std_shared = defaultdict(list)
            for block_id in range(4):
                results_mean_shared[block_id] = np.mean([results_mean[n_id] for n_id in range(block_id,block_id+128)], axis = 0)
                results_std_shared[block_id] = np.mean([results_std[n_id] for n_id in range(block_id,block_id+128)], axis = 0)

        if type(parameter) is pyhalbe.HICANN.shared_parameter:
            all_steps = self.get_shared_steps()
            for block_id in range(4):
                results_mean_shared[block_id] = np.array(results_mean_shared[block_id])
                results_std_shared[block_id] = np.array(results_std_shared[block_id])
                steps = [step[parameter].value for step in all_steps[block_id]]
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
                steps = [step[parameter].value for step in all_steps[neuron_id]]
                if linear_fit:
                    # linear fit
                    m, b = np.linalg.lstsq(zip(results_mean[neuron_id], [1]*len(results_mean[neuron_id])), steps)[0]
                    coeffs = [b, m]
                    # TODO find criteria for broken neurons. Until now, no broken neurons exist
                    #if parameter is pyhalbe.HICANN.neuron_parameter.E_l:
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
        md.setAuthor("pycairo")
        md.setComment("calibration")

        logger = self.logger
        
        if type(parameter) is pyhalbe.HICANN.neuron_parameter:
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

    def init_experiment(self):
        super(BaseCalibration, self).init_experiment()
        if self._calib_backend is None:
            raise TypeError("can not store results without Calibtic backend")
        if self._red_nrns is None:
            raise TypeError("can not store defects without Redman backend")
        self.repetitions = 1


class Calibrate_E_l(BaseCalibration):
    """E_l calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_E_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.I_gl: Current(1000),
                neuron_parameter.V_t: Voltage(1200),
                neuron_parameter.I_convi: DAC(1023),
                neuron_parameter.I_convx: DAC(1023),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_l, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][pyhalbe.HICANN.shared_parameter.V_reset] = Voltage(500)
        return parameters

    def get_steps(self):
        steps = []
        for voltage in range(600, 1100, 25):
            steps.append({pyhalbe.HICANN.neuron_parameter.E_l: Voltage(voltage),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_E_l, self).init_experiment()
        self.repetitions = 5
        self.save_results = True
        self.save_traces = False
        self.E_syni_dist = -100
        self.E_synx_dist = +100
        self.description = "Calibrate_E_l with Esyn set AFTER calibration."

    def process_trace(self, t, v):
        if np.std(v)*1000>50:
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_trace.p"), 'wb'))
            raise ValueError
        return np.mean(v)*1000 # Get the mean value * 1000 for mV

    def process_results(self, neuron_ids):
        super(Calibrate_E_l, self).process_calibration_results(neuron_ids, neuron_parameter.E_l, linear_fit=True)

    def store_results(self):
        super(Calibrate_E_l, self).store_calibration_results(neuron_parameter.E_l)


class Calibrate_V_t(BaseCalibration):
    """V_t calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_V_t, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                pyhalbe.HICANN.neuron_parameter.E_l: Voltage(1200, apply_calibration = True),  # TODO apply calibration?
                pyhalbe.HICANN.neuron_parameter.I_gl: Current(1000),
                pyhalbe.HICANN.neuron_parameter.I_convi: DAC(1023),
                pyhalbe.HICANN.neuron_parameter.I_convx: DAC(1023),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_V_t, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][pyhalbe.HICANN.shared_parameter.V_reset] = Voltage(500)
        return parameters

    def get_steps(self):
        steps = []
        for voltage in range(700,1175,25): 
            steps.append({pyhalbe.HICANN.neuron_parameter.V_t: Voltage(voltage)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_V_t, self).init_experiment()
        self.repetitions = 5
        self.save_results = True
        self.save_traces = False 
        self.E_syni_dist = -100
        self.E_synx_dist = +100
        self.description = "Basic Calibrate_V_t with Iconv ON and 1000 I_pl. Calibrated E_l."

    def process_trace(self, t, v):
        return np.max(v)*1000  # Get the mean value * 1000 for mV

    def process_results(self, neuron_ids):
        super(Calibrate_V_t, self).process_calibration_results(neuron_ids, neuron_parameter.V_t)

    def store_results(self):
        super(Calibrate_V_t, self).store_calibration_results(neuron_parameter.V_t)


class Calibrate_V_reset(BaseCalibration):
    """V_reset calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_V_reset, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(1100, apply_calibration=True),
                neuron_parameter.I_gl: Current(1000),
                neuron_parameter.V_t: Voltage(900, apply_calibration=True),
                #neuron_parameter.I_pl: DAC(5),  # Long refractory period
                neuron_parameter.I_convi: DAC(1023),
                neuron_parameter.I_convx: DAC(1023),
            })
        # TODO apply V_t calibration?
        return parameters

    def get_shared_steps(self):
        steps = []
        #for voltage in range(600, 900, 25):
        for voltage in range(600, 900, 100):
            steps.append({pyhalbe.HICANN.shared_parameter.V_reset: Voltage(voltage)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_V_reset, self).init_experiment()
        self.repetitions = 5
        self.save_results = True
        self.save_traces = False
        self.E_syni_dist = -100
        self.E_synx_dist = +100
        self.description = "Calibrate_V_reset, I_pl HIGH. E_l and V_t calibrated."

    def process_trace(self, t, v):
        return np.min(v)*1000  # Get the mean value * 1000 for mV

    def process_results(self, neuron_ids):
        super(Calibrate_V_reset, self).process_calibration_results(neuron_ids, shared_parameter.V_reset)

    def store_results(self):
        # TODO detect and store broken neurons
        super(Calibrate_V_reset, self).store_calibration_results(shared_parameter.V_reset)


# TODO, playground for digital spike measures atm.
class Calibrate_g_L(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_g_L, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                pyhalbe.HICANN.neuron_parameter.E_l: Voltage(1100, apply_calibration = True),
                pyhalbe.HICANN.neuron_parameter.V_t: Voltage(1000, apply_calibration = True),
                pyhalbe.HICANN.neuron_parameter.I_convx: DAC(1023),
                pyhalbe.HICANN.neuron_parameter.I_convi: DAC(1023),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_g_L, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][pyhalbe.HICANN.shared_parameter.V_reset] = Voltage(800, apply_calibration = True)
        return parameters

    def get_steps(self):
        steps = []
        #for current in range(700, 1050, 50):
        for current in range(700, 1000, 100):
            steps.append({pyhalbe.HICANN.neuron_parameter.I_gl: Current(current)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_g_L, self).init_experiment()
        self.description = "Capacitance g_L experiment." # Change this for all child classes
        self.repetitions = 1
        self.E_syni_dist = -100
        self.E_synx_dist = +100
        self.save_results = True
        self.save_traces = False

    def process_trace(self, t, v):
        spk = pycairo.logic.spikes.detect_spikes(t,v)
        f = pycairo.logic.spikes.spikes_to_freqency(spk) 
        E_l = 1100.
        C = 2.16456E-12
        V_t = max(v)*1000.
        V_r = min(v)*1000.
        g_l = f * C * np.log((V_r-E_l)/(V_t-E_l)) * 1E9
        return g_l

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
                    results_mean[neuron_id].append(np.mean(step_results[neuron_id]))
                    results_std[neuron_id].append(np.std(step_results[neuron_id]))
                repetition = 1
                step_results = defaultdict(list)

        
        all_steps = self.get_steps()
        for neuron_id in neuron_ids:
            results_mean[neuron_id] = np.array(results_mean[neuron_id])
            results_std[neuron_id] = np.array(results_std[neuron_id])
            steps = [step[parameter].value for step in all_steps[neuron_id]]
            if linear_fit:
                # linear fit
                m, b = np.linalg.lstsq(zip(results_mean[neuron_id], [1]*len(results_mean[neuron_id])), steps)[0]
                coeffs = [b, m]
                if parameter is pyhalbe.HICANN.neuron_parameter.E_l:
                    if not (m > 1.0 and m < 1.5 and b < 0 and b > -500):
                        # this neuron is broken
                        results_broken.append(neuron_id)
            else:
                # fit polynomial to results
                weight = 1./(results_std[neuron_id] + 1e-8)  # add a tiny value because std may be zero
                # note that np.polynomial.polynomial.polyfit coefficients have
                # reverse order compared to np.polyfit
                # to reverse coefficients: rcoeffs = coeffs[::-1]
                coeffs = np.polynomial.polynomial.polyfit(results_mean[neuron_id], steps, 2, w=weight)
                # TODO check coefficients for broken behavior
            results_polynomial[neuron_id] = create_pycalibtic_polynomial(coeffs)

        self.results_mean = results_mean
        self.results_std = results_std

        # make final results available
        self.results_polynomial = results_polynomial
        self.results_broken = results_broken

    def process_results(self, neuron_ids):
        self.process_calibration_results(neuron_ids, pyhalbe.HICANN.neuron_parameter.I_gl)

    def store_results(self):
        # TODO detect and store broken neurons
        super(Calibrate_g_L, self).store_calibration_results(neuron_parameter.I_gl)



# TODO, playground for digital spike measures atm.
class Calibrate_g_L_stepcurrent(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_g_L_stepcurrent, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                pyhalbe.HICANN.neuron_parameter.E_l: Voltage(900, apply_calibration = True),
                pyhalbe.HICANN.neuron_parameter.V_t: Voltage(1300, apply_calibration = True),
                pyhalbe.HICANN.neuron_parameter.I_convx: DAC(1023),
                pyhalbe.HICANN.neuron_parameter.I_convi: DAC(1023),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_g_L_stepcurrent, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][pyhalbe.HICANN.shared_parameter.V_reset] = Voltage(700, apply_calibration = True)
        return parameters

    def get_steps(self):
        steps = []
        #for current in range(700, 1050, 50):
        for current in range(700, 1000, 100):
            steps.append({pyhalbe.HICANN.neuron_parameter.I_gl: Current(current)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_g_L_stepcurrent, self).init_experiment()
        self.description = "Capacitance g_L experiment. Membrane is fitted after step current" # Change this for all child classes
        self.repetitions = 1
        self.E_syni_dist = -100
        self.E_synx_dist = +100
        self.save_results = True
        self.save_traces = False

    def prepare_measurement(self, step_parameters, step_id, rep_id):
        """Prepare measurement.
        Perform reset, write general hardware settings.
        """
        neuron_parameters = step_parameters[0]
        shared_parameters = step_parameters[1]

        fgc = pyhalbe.HICANN.FGControl()
        # Set neuron parameters for each neuron
        for neuron_id in neuron_parameters:
            coord = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(neuron_id))
            for parameter in neuron_parameters[neuron_id]:
                value = neuron_parameters[neuron_id][parameter]
                fgc.setNeuron(coord, parameter, value)
            fgstim = pyhalbe.HICANN.FGStimulus(10,10,False)
            self.sthal.hicann.setCurrentStimulus(coord, fgstim)

        # Set block parameters for each block
        for block_id in shared_parameters:
            coord = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(block_id))
            for parameter in shared_parameters[block_id]:
                value = shared_parameters[block_id][parameter]
                fgc.setShared(coord, parameter, value)

        self.sthal.hicann.floating_gates = fgc

        if self.bigcap is True:
            # TODO add bigcap functionality
            pass
        
        if self.save_results:
            if not os.path.isdir(os.path.join(self.folder,"floating_gates/")):
                os.mkdir(os.path.join(self.folder,"floating_gates/"))
            pickle.dump(fgc, open(os.path.join("floating_gates/","step{}rep{}.p".format(step_id,rep_id)), 'wb'))

        self.sthal.write_config()

    def process_trace(self, t, v):
        spk = pycairo.logic.spikes.detect_spikes(t,v)
        f = pycairo.logic.spikes.spikes_to_freqency(spk) 
        E_l = 1100.
        C = 2.16456E-12
        V_t = max(v)*1000.
        V_r = min(v)*1000.
        g_l = f * C * np.log((V_r-E_l)/(V_t-E_l)) * 1E9
        return g_l

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass






class Calibrate_tau_ref(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_tau_ref, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(800),
                neuron_parameter.V_t: Voltage(700),
                shared_parameter.V_reset: Voltage(500),
                neuron_parameter.I_gl: Current(1000),
                neuron_parameter.V_syntcx: Voltage(10)
            })
        return parameters

    def measure(self, neuron_ids):
        import pycairo.simulator
        params = self.get_parameters()
        results = {}
        base_freq = pycairo.simulator.Simulator().compute_freq(30e-6, 0, params)

        for neuron_id in neuron_ids:
            self.sthal.switch_analog_output(neuron_id)
            self.sthal.adc.record()
            ts = np.array(self.sthal.adc.getTimestamps())
            v = np.array(self.sthal.adc.trace())

            calc_freq = lambda x: (1.0 / x - 1.0 / base_freq) * 1e6  # millions?

# TODO
#            m = sorted_array_mean[sorted_array_mean != 0.0]
#            e = sorted_array_err[sorted_array_err != 0.0]
#            return calc_freq(m), calc_freq(e)

            results[neuron_id] = calc_freq(v)
            del ts

        self.all_results.append(results)


class Calibrate_tau_synx(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_tau_synx, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(600),
                neuron_parameter.E_synx: Voltage(1300),
                neuron_parameter.E_syni: Voltage(200),
                neuron_parameter.V_t: Voltage(1700),
                shared_parameter.V_reset: Voltage(500),
                neuron_parameter.I_gl: Current(1000),
            })
        return parameters

    def measure(self, neuron_ids):
        pass  # TODO


class Calibrate_tau_syni(BaseCalibration):
    def measure(self, neuron_ids):
        pass  # TODO


class Calibrate_a(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_a, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(800),
                neuron_parameter.I_gl: Current(1000),
                neuron_parameter.V_t: Voltage(700),
                neuron_parameter.I_radapt: Current(1000),
                shared_parameter.V_reset: Voltage(500),
            })
        return parameters

    def measure(self, neuron_ids):
        pass  # TODO


class Calibrate_b(BaseCalibration):
    pass

    def measure(self, neuron_ids):
        pass  # TODO


class Calibrate_tau_w(BaseCalibration):
    pass

    def measure(self, neuron_ids):
        pass  # TODO


class Calibrate_dT(BaseCalibration):
    pass

    def measure(self, neuron_ids):
        pass  # TODO


class Calibrate_V_th(BaseCalibration):
    pass

    def measure(self, neuron_ids):
        pass  # TODO
