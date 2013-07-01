"""Framework for experiments on the hardware.

Derive your custom experiment classes with functions for setup, measurement and processing
from BaseExperiment or child classes.

TODO
* source for applying calibration?
* what if source does not contain calibration data? ideal calibration?
"""

import numpy as np
from pycairo.logic.helpers import create_pycalibtic_polynomial


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
        super(DAC, self).__init__(int(value), apply_calibration)

    @classmethod
    def _check(self, value):
        if not isinstance(value, int):
            raise TypeError("DAC value is no integer")
        if value < 0 or value > 1023:
            raise ValueError("DAC value {} out of range".format(value))

    def toDAC(self):
        return self

    def __repr__(self):
        return "{}".format(self.value)


class BaseExperiment(object):
    """Base class for running experiments on individual neurons.

    Defines experiment base parameters, steps varying parameters, repetitions.
    Defines measurement procedure and postprocessing.

    Provides a function to run and process an experiment.
    """
    def __init__(self):
        self.hicann = None  # Stateful HICANN Container
        self.adc = None
        self.verbosity = 0  # print output depending on verbosity level

    def get_parameters(self):
        """Return neuron parameters for this experiment. Values can be of type Current, Voltage or DAC."""

        return {'E_l': Voltage(1000),
                'E_syni': Voltage(900),
                'E_synx': Voltage(900),
                'I_bexp': Current(0),
                'I_convi': Current(1000),
                'I_convx': Current(1000),
                'I_fire': Current(0),
                'I_gl': Current(400),
                'I_gladapt': Current(0),
                'I_intbbi': Current(2000),
                'I_intbbx': Current(2000),
                'I_pl': Current(2000),
                'I_radapt': Current(2000),
                'I_rexp': Current(750),
                'I_spikeamp': Current(2000),
                'V_exp': Voltage(536),
                'V_syni': Voltage(1000),
                'V_syntci': Voltage(900),
                'V_syntcx': Voltage(900),
                'V_synx': Voltage(1000),
                'V_t': Voltage(1000),  # recommended threshold, maximum is 1100
                'V_reset': Voltage(500)  # shared between all neurons
                }

    def prepare_parameters(self, parameters, step, neuron_ids):
        """Prepare parameters before writing them to the hardware.

        This includes converting to DAC values, applying calibration and
        merging of the step specific parameters.

        Calibration can be different for each neuron, which leads to different DAC values
        for each neuron, even though the parameter value is the same.

        Args:
            parameters: dict of all neuron parameters and values of type Current, Voltage or DAC
            step: dict of all neuron parameters different in this specific step
            neuron_ids: list of neuron ids that should get the parameters

        Returns:
            dict of neuron ids -> dict of neuron parameters -> DAC values (int)

            Example: {0: {'E_l': 400, ...},
                      1: {'E_l': 450, ...}}
        """

        neuron_step_parameters = {}
        for neuron_id in neuron_ids:
            neuron_step_parameters[neuron_id] = {}

        for p in parameters:
            if p in step:
                continue  # skip parameter
            value = parameters[p].toDAC().value
            apply_calibration = parameters[p].apply_calibration
            for neuron_id in neuron_ids:
                if apply_calibration:
                    # apply calibration
                    calibrated_value = value  # TODO
                    neuron_step_parameters[neuron_id][p] = calibrated_value
                else:
                    neuron_step_parameters[neuron_id][p] = value

        for p in step:
            value = step[p].toDAC().value
            apply_calibration = step[p].apply_calibration
            for neuron_id in neuron_ids:
                if apply_calibration:
                    # apply calibration
                    calibrated_value = value  # TODO
                    neuron_step_parameters[neuron_id][p] = calibrated_value
                else:
                    neuron_step_parameters[neuron_id][p] = value

        return neuron_step_parameters

    def prepare_measurement(self, neuron_parameters):
        """Prepare measurement.
        Perform reset, write general hardware settings.
        """
        self.hicann.reset()
        self.hicann.configure()
        self.hicann.program_fg(neuron_parameters)

    def get_repetitions(self):
        """How many times should each step be repeated?"""
        return 1

    def get_steps(self):
        """Measurement steps for sweeping.

        Returns:
            list of parameter dicts for each step

            Example: [{'E_l': Voltage(400)}, {'E_l': Voltage(600)}, {'E_l': Voltage(800)}]
        """
        return [{}]  # single step using default parameters

    def get_neurons(self):
        """Which neurons should this experiment run on?

        All neurons will be prepared with the same parameters (except for calibration differences)
        and each neuron will be measured in the measure step.
        """

        return [0]  # only neuron 0

    def run_experiment(self):
        """Run the experiment and process results."""
        parameters = self.get_parameters()
        neuron_ids = self.get_neurons()
        self.all_results = []
        for step in self.get_steps():
            if self.verbosity > 0:
                print "step {}".format(step)
            step_parameters = self.prepare_parameters(parameters, step, neuron_ids)
            step_results = []
            for r in range(self.get_repetitions()):
                if self.verbosity > 0:
                    print "repetition {}".format(r)
                self.prepare_measurement(step_parameters)
                result = self.measure(step, neuron_ids)
                step_results.append(result)
            self.all_results.append(step_results)
        if self.verbosity > 0:
            print "processing results"
        self.process_results(neuron_ids)

    def measure(self, step, neuron_ids):
        """Perform measurement(s) for a single step on one or multiple neurons."""
        results = {}
        for neuron_id in neuron_ids:
            self.hicann.activate_neuron(neuron_id)
            self.hicann.enable_analog_output(neuron_id)
            t, v = self.adc.measure_adc(1000)
            results[neuron_id] = v
        return results

    def process_results(self, neuron_ids):
        """Process measured data."""
        pass  # no processing


class Calibrate_E_l(BaseExperiment):
    def get_parameters(self):
        parameters = super(Calibrate_E_l, self).get_parameters()
        parameters.update({
            'I_gl': Current(1000),
            'V_t': Voltage(1700)
        })
        return parameters

    def get_steps(self):
        return [
            {'E_l': Voltage(300)},
            {'E_l': Voltage(400)},
            {'E_l': Voltage(500)},
            {'E_l': Voltage(600)},
            {'E_l': Voltage(700)},
            {'E_l': Voltage(800)}
        ]

    def get_repetitions(self):
        return 3

    def measure(self, step, neuron_ids):
        results = {}
        for neuron_id in neuron_ids:
            self.hicann.activate_neuron(neuron_id)
            self.hicann.enable_analog_output(neuron_id)
            t, v = self.adc.measure_adc(1000)
            E_l = np.mean(v)*1000  # multiply by 1000 for mV
            results[neuron_id] = E_l
        return results

    def process_results(self, neuron_ids):
        results_mean = {}
        results_std = {}
        results_polynomial = {}
        for neuron_id in neuron_ids:
            results_mean[neuron_id] = []
            results_std[neuron_id] = []

        for step in self.all_results:
            for neuron_id in neuron_ids:
                single_neuron_results = [measurement[neuron_id] for measurement in step]
                results_mean[neuron_id].append(np.mean(single_neuron_results))
                results_std[neuron_id].append(np.std(single_neuron_results))

        for neuron_id in neuron_ids:
            results_mean[neuron_id] = np.array(results_mean[neuron_id])
            results_std[neuron_id] = np.array(results_std[neuron_id])
            steps = [step['E_l'].value for step in self.get_steps()]
            weight = 1./results_std[neuron_id]
            # note that np.polynomial.polynomial.polyfit coefficients have
            # reverse order compared to np.polyfit
            # to reverse coefficients: rcoeffs = coeffs[::-1]
            coeffs = np.polynomial.polynomial.polyfit(results_mean[neuron_id], steps, 2, w=weight)
            results_polynomial[neuron_id] = create_pycalibtic_polynomial(coeffs)

        self.results_mean = results_mean
        self.results_std = results_std
        self.results_polynomial = results_polynomial
