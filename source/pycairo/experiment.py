"""Framework for experiments on the hardware.

Derive your custom experiment classes with functions for setup,
measurement and processing from BaseExperiment or child classes.
"""

import numpy as np
import logging
from collections import defaultdict
import pyhalbe
import pycalibtic
import pysthal
from pycairo.helpers.calibtic import create_pycalibtic_polynomial
from pycairo.helpers.sthal import UpdateAnalogOutputConfigurator


class Unit(object):
    """Base class for Current, Voltage and DAC parameter values."""
    def __init__(self, value, apply_calibration=False):
        """Args:
            value: parameter value in units of child class
            apply_calibration: apply correction to this value before writing it to the hardware?
        """
        self.value = value
        self.apply_calibration = apply_calibration

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._check(value)
        self._value = value


class Current(Unit):
    """Current in nA for hardware parameters."""
    def __init__(self, value, apply_calibration=False):
        super(Current, self).__init__(float(value), apply_calibration)

    @classmethod
    def _check(self, value):
        if value < 0. or value > 2500.:
            raise ValueError("Current value {} nA out of range".format(value))

    def toDAC(self):
        return DAC(self.value/2500.*1023., self.apply_calibration)

    def __repr__(self):
        return "{} nA".format(self.value)


class Voltage(Unit):
    """Voltage in mV for hardware parameters."""
    def __init__(self, value, apply_calibration=False):
        super(Voltage, self).__init__(float(value), apply_calibration)

    @classmethod
    def _check(self, value):
        if value < 0. or value > 1800.:
            raise ValueError("Voltage value {} mV out of range".format(value))

    def toDAC(self):
        return DAC(self.value/1800.*1023., self.apply_calibration)

    def __repr__(self):
        return "{} mV".format(self.value)


class DAC(Unit):
    def __init__(self, value, apply_calibration=False):
        super(DAC, self).__init__(int(round(value)), apply_calibration)

    @classmethod
    def _check(self, value):
        if not isinstance(value, int):
            raise TypeError("DAC value is no integer")
        if value < 0 or value > 1023:
            raise ValueError("DAC value {} out of range".format(value))

    def toDAC(self):
        return self

    def __repr__(self):
        return "{} (DAC)".format(self.value)


class BaseExperiment(object):
    """Base class for running experiments on individual neurons.

    Defines experiment base parameters, steps varying parameters, repetitions.
    Defines measurement procedure and postprocessing.

    Provides a function to run and process an experiment.
    """
    def __init__(self, neuron_ids,
                 coord_wafer=pyhalbe.Coordinate.Wafer(),
                 coord_hicann=pyhalbe.Coordinate.HICANNOnWafer(pyhalbe.geometry.Enum(0)),
                 calibtic_backend=None):

        wafer = pysthal.Wafer(coord_wafer)  # Stateful HICANN Container
        wafer.allocateHICANN(coord_hicann)
        hicann = wafer[coord_hicann]

        wafer.connect(pysthal.MagicHardwareDatabase())

        # analogRecorder() MUST be called after wafer.connect()
        adc = hicann.analogRecorder(pyhalbe.Coordinate.AnalogOnHICANN(0))
        adc.setReadoutTime(0.01)

        self.wafer = wafer
        self.hicann = hicann
        self.adc = adc
        self.cfg = pysthal.HICANNConfigurator()
        self.cfg_analog = UpdateAnalogOutputConfigurator()

        self.neuron_ids = neuron_ids
        self._repetitions = 1

        self._calib_backend = None
        self._calib_nc = None

        if calibtic_backend:
            self._calib_backend = calibtic_backend
            self.init_calibration()

    def init_experiment(self):
        """Hook for child classes. Executed by run_experiment()."""
        pass

    def init_calibration(self):
        """Initialize Calibtic backend, load existing calibration data."""

        nc = pycalibtic.NeuronCollection()
        md = pycalibtic.MetaData()
        try:  # TODO replace by 'if backend.exists("cairo")' in the future, when this function is written
            self._calib_backend.load("cairo", md, nc)
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

        if not metadata:
            metadata = self._calib_md
        self._calib_backend.store("cairo", metadata, self._calib_nc)

    def get_parameters(self):
        """Return neuron parameters for this experiment. Values can be of type Current, Voltage or DAC.

        Returns:
            dict of neuron id -> dict of parameters -> values
        """

        # use halbe default parameters
        coord_neuron = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(0))
        coord_block = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(0))
        fgc = pyhalbe.HICANN.FGControl()

        result = {}
        for name, param in pyhalbe.HICANN.neuron_parameter.names:
            if name[0] in ("E", "V"):
                result[param] = Voltage(fgc.getNeuron(coord_neuron, param))
            elif name[0] == "I":
                result[param] = Current(fgc.getNeuron(coord_neuron, param))
            else:  # "__last_neuron"
                pass

        # add V_reset
        param = pyhalbe.HICANN.shared_parameter.V_reset
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

            Example: {0: {pyhalbe.HICANN.neuron_parameter.E_l: 400, ...},
                      1: {pyhalbe.HICANN.neuron_parameter.E_l: 450, ...}}
        """

        steps = self.get_steps()
        neuron_ids = self.get_neurons()

        step_parameters = self.get_parameters()
        for neuron_id in neuron_ids:
            step_parameters[neuron_id].update(steps[neuron_id][step_id])
            for param in step_parameters[neuron_id]:
                # convert to DAC and apply calibration
                value = step_parameters[neuron_id][param].toDAC().value
                apply_calibration = step_parameters[neuron_id][param].apply_calibration
                if apply_calibration:
                    # apply calibration
                    ncal = self._calib_nc.at(neuron_id)
                    calibration = ncal.at(param)
                    calibrated_value = calibration.apply(value)
                    step_parameters[neuron_id][param] = calibrated_value
                else:
                    step_parameters[neuron_id][param] = value

        return step_parameters

    def prepare_measurement(self, neuron_parameters):
        """Prepare measurement.
        Perform reset, write general hardware settings.
        """

        for neuron_id in neuron_parameters:
            coord = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.geometry.Enum(neuron_id))
            neuron = self.hicann.neurons[coord]
            # use fastest membrane possible
            neuron.bigcap = True
            neuron.fast_I_gl = True
            neuron.slow_I_gl = False

        fgc = pyhalbe.HICANN.FGControl()
        V_reset = None
        for neuron_id in neuron_parameters:
            coord = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.geometry.Enum(neuron_id))
            for parameter in neuron_parameters[neuron_id]:
                if parameter is pyhalbe.HICANN.shared_parameter.V_reset:
                    V_reset = neuron_parameters[neuron_id][parameter]
                    # maybe make sure that all neurons have the same value for
                    # V_reset here?
                    continue
                value = neuron_parameters[neuron_id][parameter]
                fgc.setNeuron(coord, parameter, value)

        if V_reset:
            for block in range(4):
                coord = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(block))
                fgc.setShared(coord, pyhalbe.HICANN.shared_parameter.V_reset, V_reset)

        self.hicann.floating_gates = fgc
        self.wafer.configure(self.cfg)

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

            Example: [
                        { # first step
                            # neuron 0
                            0: {pyhalbe.HICANN.neuron_parameter.E_l: Voltage(400)},
                            1: {pyhalbe.HICANN.neuron_parameter.E_l: Voltage(400)},
                        },
                        { # second step
                            0: pyhalbe.HICANN.neuron_parameter.E_l: Voltage(600)},
                            1: pyhalbe.HICANN.neuron_parameter.E_l: Voltage(650)},
                        },
                            0: {pyhalbe.HICANN.neuron_parameter.E_l: Voltage(800)}
                            1: {pyhalbe.HICANN.neuron_parameter.E_l: Voltage(800)}
                        }
                    ]
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

        self.init_experiment()
        neuron_ids = self.get_neurons()
        #parameters = self.get_parameters()

        self.all_results = []
        steps = self.get_steps()
        num_steps = len(steps[neuron_ids[0]])
        for step_id in range(num_steps):
            logging.info("step {}".format(step_id))
            step_parameters = self.prepare_parameters(step_id)
            for r in range(self.repetitions):
                logging.info("repetition {}".format(r))
                self.prepare_measurement(step_parameters)
                self.measure(neuron_ids)
        logging.info("processing results")
        self.process_results(neuron_ids)
        self.store_results()

    def measure(self, neuron_ids):
        """Perform measurements for a single step on one or multiple neurons."""
        results = {}
        for neuron_id in neuron_ids:
            coord_neuron = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(neuron_id))
            self.hicann.enable_l1_output(coord_neuron, pyhalbe.HICANN.L1Address(0))
            self.hicann.enable_aout(coord_neuron, pyhalbe.Coordinate.AnalogOnHICANN(0))
            self.wafer.configure(self.cfg)
            v = self.adc.read()
            results[neuron_id] = v
        self.all_results.append[results]

    def process_results(self, neuron_ids):
        """Process measured data."""
        pass  # no processing

    def store_results(self):
        """Hook for storing results in child classes."""
        pass


class BaseCalibration(BaseExperiment):
    """Base class for calibration experiments."""
    def get_parameters(self):
        parameters = {pyhalbe.HICANN.neuron_parameter.E_l: Voltage(1000),
                      pyhalbe.HICANN.neuron_parameter.E_syni: Voltage(900),
                      pyhalbe.HICANN.neuron_parameter.E_synx: Voltage(900),
                      pyhalbe.HICANN.neuron_parameter.I_bexp: Current(0),
                      pyhalbe.HICANN.neuron_parameter.I_convi: Current(1000),
                      pyhalbe.HICANN.neuron_parameter.I_convx: Current(1000),
                      pyhalbe.HICANN.neuron_parameter.I_fire: Current(0),
                      pyhalbe.HICANN.neuron_parameter.I_gl: Current(400),
                      pyhalbe.HICANN.neuron_parameter.I_gladapt: Current(0),
                      pyhalbe.HICANN.neuron_parameter.I_intbbi: Current(2000),
                      pyhalbe.HICANN.neuron_parameter.I_intbbx: Current(2000),
                      pyhalbe.HICANN.neuron_parameter.I_pl: Current(2000),
                      pyhalbe.HICANN.neuron_parameter.I_radapt: Current(2000),
                      pyhalbe.HICANN.neuron_parameter.I_rexp: Current(750),
                      pyhalbe.HICANN.neuron_parameter.I_spikeamp: Current(2000),
                      pyhalbe.HICANN.neuron_parameter.V_exp: Voltage(536),
                      pyhalbe.HICANN.neuron_parameter.V_syni: Voltage(1000),
                      pyhalbe.HICANN.neuron_parameter.V_syntci: Voltage(900),
                      pyhalbe.HICANN.neuron_parameter.V_syntcx: Voltage(900),
                      pyhalbe.HICANN.neuron_parameter.V_synx: Voltage(1000),
                      pyhalbe.HICANN.neuron_parameter.V_t: Voltage(1000),  # recommended threshold, maximum is 1100
                      pyhalbe.HICANN.shared_parameter.V_reset: Voltage(500)
                      }
        return defaultdict(lambda: parameters)

    def process_calibration_results(self, neuron_ids, parameter):
        """This base class function can be used by child classes as process_results."""
        # containers for final results
        results_mean = defaultdict(list)
        results_std = defaultdict(list)
        results_polynomial = {}

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

        # fit polynomial to results
        all_steps = self.get_steps()
        for neuron_id in neuron_ids:
            results_mean[neuron_id] = np.array(results_mean[neuron_id])
            results_std[neuron_id] = np.array(results_std[neuron_id])
            steps = [step[parameter].value for step in all_steps[neuron_id]]
            weight = 1./results_std[neuron_id]
            # note that np.polynomial.polynomial.polyfit coefficients have
            # reverse order compared to np.polyfit
            # to reverse coefficients: rcoeffs = coeffs[::-1]
            coeffs = np.polynomial.polynomial.polyfit(results_mean[neuron_id], steps, 2, w=weight)
            results_polynomial[neuron_id] = create_pycalibtic_polynomial(coeffs)

        # make final results available
        self.results_mean = results_mean
        self.results_std = results_std
        self.results_polynomial = results_polynomial

    def store_calibration_results(self, parameter):
        """This base class function can be used by child classes as store_results."""
        results = self.results_polynomial
        md = pycalibtic.MetaData()
        md.setAuthor("pycairo")
        md.setComment("calibration")
        nc = self._calib_nc
        for neuron_id in results:
            if not nc.exists(neuron_id):
                logging.info("no existing calibration data for neuron {} found, creating default dataset")
                ncal = pycalibtic.NeuronCalibration()
                nc.insert(neuron_id, ncal)
            nc.at(neuron_id).reset(parameter, results[neuron_id])

        self.store_calibration(md)

    def init_experiment(self):
        if not self._calib_backend:
            raise TypeError("can not store results without Calibtic backend")
        self.repetitions = 3


class Calibrate_E_l(BaseCalibration):
    """E_l calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_E_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                pyhalbe.HICANN.neuron_parameter.I_gl: Current(1000),
                pyhalbe.HICANN.neuron_parameter.V_t: Voltage(1700)
            })
        return parameters

    def get_steps(self):
        steps = []
        for voltage in range(300, 900, 100):
            steps.append({pyhalbe.HICANN.neuron_parameter.E_l: Voltage(voltage)})
        return defaultdict(lambda: steps)

    def measure(self, neuron_ids):
        results = {}
        for neuron_id in neuron_ids:
            coord_neuron = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(neuron_id))
            self.hicann.enable_l1_output(coord_neuron, pyhalbe.HICANN.L1Address(0))
            self.hicann.enable_aout(coord_neuron, pyhalbe.Coordinate.AnalogOnHICANN(0))
            self.wafer.configure(self.cfg_analog)
            v = self.adc.read()
            E_l = np.mean(v)*1000  # multiply by 1000 for mV
            results[neuron_id] = E_l
        self.all_results.append(results)

    def process_results(self, neuron_ids):
        super(Calibrate_E_l, self).process_calibration_results(neuron_ids, pyhalbe.HICANN.neuron_parameter.E_l)

    def store_results(self):
        # TODO sanity check on self.results_polynomial
        # before storing anything
        super(Calibrate_E_l, self).store_calibration_results(pyhalbe.HICANN.neuron_parameter.E_l)


class Calibrate_V_t(BaseCalibration):
    """V_t calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_V_t, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                pyhalbe.HICANN.neuron_parameter.E_l: Voltage(1100),
                pyhalbe.HICANN.neuron_parameter.I_gl: Current(1000)
            })
        return parameters

    def get_steps(self):
        steps = []
        for voltage in (600, 700, 800):
            steps.append({pyhalbe.HICANN.neuron_parameter.V_t: Voltage(voltage)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_V_t, self).init_experiment()
        self.repetitions = 2

    def measure(self, neuron_ids):
        results = {}
        for neuron_id in neuron_ids:
            coord_neuron = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(neuron_id))
            self.hicann.enable_l1_output(coord_neuron, pyhalbe.HICANN.L1Address(0))
            self.hicann.enable_aout(coord_neuron, pyhalbe.Coordinate.AnalogOnHICANN(0))
            v = self.adc.read()
            V_t = np.max(v)*1000  # multiply by 1000 for mV
            results[neuron_id] = V_t
        self.all_results.append(results)

    def process_results(self, neuron_ids):
        super(Calibrate_V_t, self).process_calibration_results(neuron_ids, pyhalbe.HICANN.neuron_parameter.V_t)

    def store_results(self):
        super(Calibrate_V_t, self).store_calibration_results(pyhalbe.HICANN.neuron_parameter.V_t)


class Calibrate_V_reset(BaseCalibration):
    """V_reset calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_V_reset, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                pyhalbe.HICANN.neuron_parameter.E_l: Voltage(1100),
                pyhalbe.HICANN.neuron_parameter.I_gl: Current(1000)
            })
        return parameters

    def get_steps(self):
        steps = []
        for voltage in (400, 500, 600):
            steps.append({pyhalbe.HICANN.neuron_parameter.V_reset: Voltage(voltage)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_V_reset, self).init_experiment()
        self.repetitions = 2

    def measure(self, neuron_ids):
        results = {}
        for neuron_id in neuron_ids:
            coord_neuron = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(neuron_id))
            self.hicann.enable_l1_output(coord_neuron, pyhalbe.HICANN.L1Address(0))
            self.hicann.enable_aout(coord_neuron, pyhalbe.Coordinate.AnalogOnHICANN(0))
            v = self.adc.read()
            V_reset = np.min(v)*1000  # multiply by 1000 for mV
            results[neuron_id] = V_reset
        self.all_results.append(results)

    def process_results(self, neuron_ids):
        super(Calibrate_V_reset, self).process_calibration_results(neuron_ids, pyhalbe.HICANN.neuron_parameter.V_reset)

    def store_results(self):
        super(Calibrate_V_reset, self).store_calibration_results(pyhalbe.HICANN.neuron_parameter.V_reset)


class Calibrate_g_L(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_g_L, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                pyhalbe.HICANN.neuron_parameter.E_l: Voltage(800),
                pyhalbe.HICANN.neuron_parameter.V_t: Voltage(700),
                pyhalbe.HICANN.neuron_parameter.V_reset: Voltage(500),
            })
        return parameters


class Calibrate_tau_ref(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_tau_ref, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                pyhalbe.HICANN.neuron_parameter.E_l: Voltage(800),
                pyhalbe.HICANN.neuron_parameter.V_t: Voltage(700),
                pyhalbe.HICANN.neuron_parameter.V_reset: Voltage(500),
                pyhalbe.HICANN.neuron_parameter.g_L: Current(1000),
            })
        return parameters


class Calibrate_tau_synx(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_tau_synx, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                pyhalbe.HICANN.neuron_parameter.E_l: Voltage(600),
                pyhalbe.HICANN.neuron_parameter.E_synx: Voltage(1300),
                pyhalbe.HICANN.neuron_parameter.E_syni: Voltage(200),
                pyhalbe.HICANN.neuron_parameter.V_t: Voltage(1700),
                pyhalbe.HICANN.neuron_parameter.V_reset: Voltage(500),
                pyhalbe.HICANN.neuron_parameter.g_L: Current(1000),
            })
        return parameters


class Calibrate_tau_syni(BaseCalibration):
    pass


class Calibrate_a(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_a, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                pyhalbe.HICANN.neuron_parameter.E_l: Voltage(800),
                pyhalbe.HICANN.neuron_parameter.g_L: Current(1000),
                pyhalbe.HICANN.neuron_parameter.V_t: Voltage(700),
                pyhalbe.HICANN.neuron_parameter.tau_w: Current(1000),  # TODO Current?
                pyhalbe.HICANN.neuron_parameter.V_reset: Voltage(500),
            })
        return parameters


class Calibrate_b(BaseCalibration):
    pass


class Calibrate_tau_w(BaseCalibration):
    pass


class Calibrate_dT(BaseCalibration):
    pass


class Calibrate_V_th(BaseCalibration):
    pass
