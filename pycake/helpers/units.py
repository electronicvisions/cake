"""Conversion of Voltage/Current to DAC"""

import numpy

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

    @staticmethod
    def _check(value):
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

    @staticmethod
    def _check(value):
        if value < 0. or value > 1800.:
            raise ValueError("Voltage value {} mV out of range".format(value))

    def toDAC(self):
        return DAC(self.value/1800.*1023., self.apply_calibration)

    def __repr__(self):
        return "{} mV".format(self.value)


class DAC(Unit):
    def __init__(self, value, apply_calibration=False):
        super(DAC, self).__init__(int(round(value)), apply_calibration)

    @staticmethod
    def _check(self, value):
        if not isinstance(value, int):
            raise TypeError("DAC value is no integer")
        if value < 0 or value > 1023:
            raise ValueError("DAC value {} out of range".format(value))

    def toDAC(self):
        return self

    def __repr__(self):
        return "{} (DAC)".format(self.value)

def linspace_voltage(start, end, steps, apply_calibration=False):
    """generates a numpy.linspace of voltage values"""
    return [Voltage(step, apply_calibration)
            for step in numpy.linspace(start, end, steps)]

def linspace_current(start, end, steps, apply_calibration=False):
    """generates a numpy.linspace of current values"""
    return [Current(step, apply_calibration)
            for step in numpy.linspace(start, end, steps)]
