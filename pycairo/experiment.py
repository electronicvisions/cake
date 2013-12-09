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
from pycairo.helpers.calibtic import create_pycalibtic_polynomial
from pycairo.logic import spikes
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
        self._calib_nc = None

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
        self.description = "Basic experiment."  # Change this for all child classes
        localtime = time.localtime()
        self.folder = "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)

    def init_redman(self, backend):
        """Initialize defect management for given backend."""
        # FIXME default coordinates
        coord_hglobal = self.sthal.index()  # grab HICANNGlobal from StHAL
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

        nc = pycalibtic.NeuronCollection()
        md = pycalibtic.MetaData()

        # grab Coordinate.HICANNGlobal from StHAL
        c_hg = self.sthal.index()

        # collection should be named "w<wafer-id>-h<hicann-id>"
        name = "w{}-h{}".format(int(c_hg.wafer().id()), int(c_hg.on_wafer().id()))
        try:  # TODO replace by 'if backend.exists()' in the future, when this function is written
            self._calib_backend.load(name, md, nc)
        except RuntimeError, e:
            if e.message != "data set not found":
                raise RuntimeError(e)
            else:
                # backend does not exist
                pass
        self._calib_nc = nc
        self._calib_md = md

    def store_calibration(self, metadata=None):
        """Write calibration data to backend"""
        self._red_hicann.commit()

        if not metadata:
            metadata = self._calib_md
        self._calib_backend.store("cairo", metadata, self._calib_nc)

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

        # add V_reset
        param = shared_parameter.V_reset
        result[param] = fgc.getShared(coord_block, param)

        return defaultdict(lambda: result)

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
        neuron_ids = self.get_neurons()

        step_parameters = self.get_parameters()

        # Give info about nonexisting calibrations:
        for param in step_parameters[0]:
            try:
                ncal = self._calib_nc.at(0).at(param)
            except (RuntimeError, IndexError):
                self.logger.WARN("No calibration found for {}. Using uncalibrated values.".format(param))

        # FIXME properly handle broken neurons here?
        for neuron_id in neuron_ids:
            step_parameters[neuron_id].update(steps[neuron_id][step_id])
            coord_neuron = Coordinate.NeuronOnHICANN(Enum(neuron_id))
            broken = not self._red_nrns.has(coord_neuron)
            for param in step_parameters[neuron_id]:
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

            # Set E_syni and E_synx AFTER calibration
            if self.E_syni_dist and self.E_synx_dist:
                print "*********************"
                E_l = step_parameters[neuron_id][neuron_parameter.E_l]
                i_dist = int(round(self.E_syni_dist * 1023./1800.))  # Convert mV to DAC
                x_dist = int(round(self.E_synx_dist * 1023./1800.))  # Convert mV to DAC
                step_parameters[neuron_id][neuron_parameter.E_syni] = type(step_parameters[neuron_id][pyhalbe.HICANN.neuron_parameter.E_syni])(E_l + i_dist)
                step_parameters[neuron_id][neuron_parameter.E_synx] = type(step_parameters[neuron_id][pyhalbe.HICANN.neuron_parameter.E_synx])(E_l + x_dist)

        return step_parameters

    def prepare_measurement(self, neuron_parameters, step_id, rep_id):
        """Prepare measurement.
        Perform reset, write general hardware settings.
        """

        for neuron_id in neuron_parameters:
            coord = Coordinate.NeuronOnHICANN(Enum(neuron_id))
            neuron = self.sthal.hicann.neurons[coord]
            # use fastest membrane possible
            neuron.bigcap = True
            neuron.fast_I_gl = True
            neuron.slow_I_gl = False

        fgc = pyhalbe.HICANN.FGControl()
        V_reset = None
        for neuron_id in neuron_parameters:
            coord = Coordinate.NeuronOnHICANN(Enum(neuron_id))
            for parameter in neuron_parameters[neuron_id]:
                if parameter is shared_parameter.V_reset:
                    V_reset = neuron_parameters[neuron_id][parameter]
                    # maybe make sure that all neurons have the same value for
                    # V_reset here?
                    continue
                value = neuron_parameters[neuron_id][parameter]
                fgc.setNeuron(coord, parameter, value)

        if V_reset:
            for block in range(4):
                coord = Coordinate.FGBlockOnHICANN(Enum(block))
                fgc.setShared(coord, shared_parameter.V_reset, V_reset)

        self.sthal.hicann.floating_gates = fgc

        if self.save_results:
            if not os.path.isdir(os.path.join(self.folder, "floating_gates")):
                os.mkdir(os.path.join(self.folder, "floating_gates"))
            pickle.dump(fgc, open("{}/floating_gates/step{}rep{}.p".format(self.folder, step_id, rep_id), 'wb'))
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

        self.all_results = []
        steps = self.get_steps()
        num_steps = len(steps[neuron_ids[0]])

        # Save default and step parameters and description to files
        # Also save sthal container to extract standard sthal parameters later:

        if self.save_results:
            if not os.path.isdir(self.folder):
                os.mkdir(self.folder)
            pickle.dump(self.sthal.hicann, open('{}/sthalcontainer.p'.format(self.folder), "wb"))
            paramdump = {nid: {parameters[nid].keys()[pid].name: parameters[nid].values()[pid] for pid in parameters[0].keys()} for nid in self.get_neurons()}
            pickle.dump(paramdump[0], open("{}/parameters.p".format(self.folder), "wb"))
            stepdump = [{pid.name: steps[0][sid][pid] for pid in steps[0][sid].keys()} for sid in range(num_steps)]
            pickle.dump(stepdump, open("{}/steps.p".format(self.folder), "wb"))
            pickle.dump(self.repetitions, open("{}/repetitions.p".format(self.folder), 'wb'))
            open('{}/description.txt'.format(self.folder), 'w').write(self.description)

        for step_id in range(num_steps):
            step_parameters = self.prepare_parameters(step_id)
            for r in range(self.repetitions):
                pylogging.set_loglevel(self.logger, pylogging.LogLevel.INFO)
                logger.INFO("step {} repetition {}".format(step_id, r))
                self.prepare_measurement(step_parameters, step_id, r)
                self.measure(neuron_ids, step_id, r)
        logger.INFO("processing results")
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
                if not os.path.isdir("{}/traces/".format(self.folder)):
                    os.mkdir("{}/traces/".format(self.folder))
                if not os.path.isdir("{}/traces/step{}rep{}/".format(self.folder, step_id, rep_id)):
                    os.mkdir("{}/traces/step{}rep{}/".format(self.folder, step_id, rep_id))
                pickle.dump([t, v], open("{}/traces/step{}rep{}/neuron_{}.p".format(self.folder, step_id, rep_id, neuron_id), 'wb'))

            results[neuron_id] = self.process_trace(t, v)
        # Now store measurements in a file:
        if self.save_results:
            if not os.path.isdir("{}/results/".format(self.folder)):
                os.mkdir("{}/results/".format(self.folder))
            pickle.dump(results, open("{}/results/step{}_rep{}.p".format(self.folder, step_id, rep_id), 'wb'))
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
        parameters = {neuron_parameter.E_l: Voltage(1000),
                      neuron_parameter.E_syni: Voltage(900),
                      neuron_parameter.E_synx: Voltage(1100),
                      neuron_parameter.I_bexp: Current(0),
                      neuron_parameter.I_convi: Current(1000),  # set 0
                      neuron_parameter.I_convx: Current(1000),  # set 0
                      neuron_parameter.I_fire: Current(0),
                      neuron_parameter.I_gl: Current(400),
                      neuron_parameter.I_gladapt: Current(0),
                      neuron_parameter.I_intbbi: Current(2000),
                      neuron_parameter.I_intbbx: Current(2000),
                      neuron_parameter.I_pl: Current(2000),
                      neuron_parameter.I_radapt: Current(2000),
                      neuron_parameter.I_rexp: Current(750),
                      neuron_parameter.I_spikeamp: Current(2000),
                      neuron_parameter.V_exp: Voltage(536),
                      neuron_parameter.V_syni: Voltage(1000),
                      neuron_parameter.V_syntci: Voltage(900),
                      neuron_parameter.V_syntcx: Voltage(900),
                      neuron_parameter.V_synx: Voltage(1000),
                      neuron_parameter.V_t: Voltage(1000),  # recommended threshold, maximum is 1100
                      shared_parameter.V_reset: Voltage(500)
                      }
        return defaultdict(lambda: dict(parameters))

    def process_calibration_results(self, neuron_ids, parameter, linear_fit=False):
        """This base class function can be used by child classes as process_results."""
        # containers for final results
        results_mean = defaultdict(list)
        results_std = defaultdict(list)
        results_polynomial = {}
        results_broken = []  # will contain neuron_ids which are broken

        # self.all_results contains all measurements, need to untangle
        # repetitions and steps
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
                if parameter is neuron_parameter.E_l:
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

        # make final results available
        self.results_mean = results_mean
        self.results_std = results_std
        self.results_polynomial = results_polynomial
        self.results_broken = results_broken

    def store_calibration_results(self, parameter):
        """This base class function can be used by child classes as store_results."""
        results = self.results_polynomial
        md = pycalibtic.MetaData()
        md.setAuthor("pycairo")
        md.setComment("calibration")

        logger = self.logger

        nc = self._calib_nc
        nrns = self._red_nrns
        for neuron_id in results:
            coord_neuron = Coordinate.NeuronOnHICANN(Enum(neuron_id))
            broken = neuron_id in self.results_broken
            if broken:
                if nrns.has(coord_neuron):  # not disabled
                    nrns.disable(coord_neuron)
                # TODO reset/delete calibtic function for this neuron
            else:  # store in calibtic
                if not nc.exists(neuron_id):
                    logger.INFO("no existing calibration data for neuron {} found, creating default dataset".format(neuron_id))
                    ncal = pycalibtic.NeuronCalibration()
                    nc.insert(neuron_id, ncal)
                nc.at(neuron_id).reset(parameter, results[neuron_id])

                if type(parameter) is neuron_parameter:
                    nc.at(neuron_id).reset(parameter, results[neuron_id])
                elif type(parameter) is shared_parameter and False:
                    pass  # not implemented
                else:
                    raise NotImplementedError("parameters of this type are not supported")
        self.store_calibration(md)

    def init_experiment(self):
        super(BaseCalibration, self).init_experiment()
        if self._calib_backend is None:
            raise TypeError("can not store results without Calibtic backend")
        if self._red_nrns is None:
            raise TypeError("can not store defects without Redman backend")
        self.repetitions = 3


class Calibrate_E_l(BaseCalibration):
    """E_l calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_E_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.I_gl: Current(1000),
                neuron_parameter.V_t: Voltage(1700),
                neuron_parameter.I_convi: DAC(1023),
                neuron_parameter.I_convx: DAC(1023),
            })
        return parameters

    def get_steps(self):
        steps = []
        for voltage in range(600, 1000, 50):
            steps.append({neuron_parameter.E_l: Voltage(voltage)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_E_l, self).init_experiment()
        self.repetitions = 2
        self.save_results = True
        self.save_traces = False
        self.E_syni_dist = -100
        self.E_synx_dist = +100
        localtime = time.localtime()
        self.folder = "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
        self.description = "Calibrate_E_l with Esyn set AFTER calibration."

    def process_trace(self, t, v):
        return np.mean(v)*1000  # Get the mean value * 1000 for mV

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
                neuron_parameter.E_l: Voltage(1200, apply_calibration=True),
                shared_parameter.V_reset: Voltage(400),
                neuron_parameter.I_gl: Current(1000),
                neuron_parameter.I_convi: DAC(1023),
                neuron_parameter.I_convx: DAC(1023),
            })
        return parameters

    def get_steps(self):
        steps = []
        for voltage in range(700, 1200, 25):
            steps.append({neuron_parameter.V_t: Voltage(voltage)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_V_t, self).init_experiment()
        self.repetitions = 4
        self.save_results = True
        self.save_traces = False
        localtime = time.localtime()
        self.folder = "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
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

    def get_steps(self):
        steps = []
        for voltage in range(500, 850, 25):
            steps.append({shared_parameter.V_reset: Voltage(voltage)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_V_reset, self).init_experiment()
        self.repetitions = 4
        self.save_results = True
        self.save_traces = False
        localtime = time.localtime()
        self.folder = "exp{0:02d}{1:02d}_{2:02d}{3:02d}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
        self.description = "Basic Calibrate_V_reset, I_pl HIGH. E_l and V_t calibrated."

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
                neuron_parameter.E_l: Voltage(1100, apply_calibration=True),
                neuron_parameter.V_t: Voltage(700, apply_calibration=True),
                shared_parameter.V_reset: Voltage(500, apply_calibration=True),
                neuron_parameter.I_convx: DAC(1023),
                neuron_parameter.I_convi: DAC(1023),
            })
        return parameters

    def get_steps(self):
        steps = []
        for current in (700, 800, 900, 1000, 1100):
            steps.append({neuron_parameter.I_gl: Current(current)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_g_L, self).init_experiment()
        self.description = "Capacitance g_L experiment."
        localtime = time.localtime()
        self.folder = "exp{}{}_{}{}".format(localtime.tm_mon, localtime.tm_mday, localtime.tm_hour, localtime.tm_min)
        self.repetitions = 2
        self.save_results = True
        self.save_traces = True

    def process_trace(self, t, v):
        detected_spikes = spikes.detect_spikes(t, v)
        freq = spikes.spikes_to_freqency(detected_spikes)
        return freq

    def process_results(self, neuron_ids):
        super(Calibrate_g_L, self).process_calibration_results(neuron_ids, neuron_parameter.I_gl)

    def store_results(self):
        # TODO detect and store broken neurons
        super(Calibrate_g_L, self).store_calibration_results(neuron_parameter.I_gl)


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
