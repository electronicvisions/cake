"""Framework for experiments on the hardware.

Derive your custom experiment classes with functions for setup, measurement and processing
from BaseExperiment or child classes.

TODO
* source for applying calibration?
* what if source does not contain calibration data? ideal calibration?
"""


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
    def _check(value):
        if value < 0. or value > 2500.:
            raise ValueError("Current value {} out of range".format(value))

    def toDAC(self):
        return DAC(self.value/2500.*1023., self.apply_calibration)


class Voltage(Unit):
    """Voltage in mV for hardware parameters."""
    def __init__(self, value, apply_calibration=False):
        super(Voltage, self).__init__(float(value), apply_calibration)

    def toDAC(self):
        return DAC(self.value/1800.*1023., self.apply_calibration)


class DAC(Unit):
    def __init__(self, value, apply_calibration=False):
        super(DAC, self).__init__(int(value), apply_calibration)

    @classmethod
    def _check(value):
        if not isinstance(value, int):
            raise TypeError("DAC value is no integer")
        if value < 0 or value > 1023:
            raise ValueError("DAC value {} out of range".format(value))

    def toDAC(self):
        return self


class BaseExperiment(object):
    def __init__(self):
        self.hicann = None  # Stateful HICANN Container
        self.adc = None

    def get_parameters(self):
        return []

    def get_parameters_to_calibrate(self):
        return []

    def measure(self):
        pass

    def prepare_parameters(self, parameters, step):
        # apply parameters
        pass

    def prepare_measurement(self, hicann_handle):
        """Prepare measurement.
        Perform reset, write general hardware settings.
        """
        pass

    def run_experiment(self, hicann_handle):
        parameters = self.get_parameters()
        for step in self.get_steps():
            step_parameters = self.prepare_parameters(parameters, step)
            for r in self.get_repetions():
                self.prepare_measurement(hicann_handle)
                self.hicann.program_fg(step_parameters)
                # TODO where does the neuron loop go?
                self.measure(step, hicann_handle)
        self.process_results()


class Calibrate_E_l(BaseExperiment):
    pass
