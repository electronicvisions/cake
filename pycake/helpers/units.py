"""Conversion of Volt/Ampere to DAC"""

import numpy

class Unit(object):
    """Base class for Current, Voltage and DAC parameter values."""
    def __init__(self, value, dac_offset=0, apply_calibration=False):
        """Args:
            value: parameter value in units of child class
            apply_calibration: apply correction to this value before writing it to the hardware?
        """
        self.value = value
        self.apply_calibration = apply_calibration
        self.dac_offset = dac_offset

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._check(value)
        self._value = value

    @staticmethod
    def _check(value):
        pass

    def toDAC(self):
        raise NotImplementedError("Unit.toDAC not implemented")

class Second(Unit):
    """ Time in s for hardware parameters"""
    def __init__(self, value, **kwargs):
        super(Second, self).__init__(float(value), **kwargs)

    @staticmethod
    def _check(value):
        if value < 0. or value > 1:
            raise ValueError("Second value {} s out of range".format(value))

    def toDAC(self):
        raise RuntimeError("Cannot convert seconds to DAC. Calibration needed.")

    def __repr__(self):
        return "{} s".format(self.value)


class Ampere(Unit):
    """Current in A for hardware parameters."""
    def __init__(self, value, **kwargs):
        super(Ampere, self).__init__(float(value), **kwargs)

    @staticmethod
    def _check(value):
        if value < 0. or value > 2.5e-6:
            raise ValueError("Current value {} A out of range".format(value))

    def toDAC(self):
        return DAC(
            self.value/2.5e-6*1023.,
            dac_offset=self.dac_offset,
            apply_calibration=self.apply_calibration)

    def __repr__(self):
        return "{} A".format(self.value)


class Volt(Unit):
    """Voltage in V for hardware parameters."""
    def __init__(self, value, **kwargs):
        super(Volt, self).__init__(float(value), **kwargs)

    @staticmethod
    def _check(value):
        if value < 0. or value > 1.8:
            raise ValueError("Volt value {} V out of range".format(value))

    def toDAC(self):
        return DAC(
            self.value/1.8*1023.,
            dac_offset=self.dac_offset,
            apply_calibration=self.apply_calibration)

    def __repr__(self):
        return "{} V".format(self.value)


class DAC(Unit):
    def __init__(self, value, **kwargs):
        super(DAC, self).__init__(int(round(value)), **kwargs)

    @staticmethod
    def _check(value):
        if not isinstance(value, int):
            raise TypeError("DAC value is no integer")
        if value < 0 or value > 1023:
            raise ValueError("DAC value {} out of range".format(value))

    def toDAC(self):
        return self

    def toAmpere(self):
        return Ampere(
            self.value/1023.*2.5e-6,
            dac_offset=self.dac_offset,
            apply_calibration=self.apply_calibration)

    def toVolt(self):
        return Volt(
            self.value/1023.*1.8,
            dac_offset=self.dac_offset,
            apply_calibration=self.apply_calibration)

    def __repr__(self):
        return "{} (DAC)".format(self.value)

def linspace_voltage(start, end, steps, apply_calibration=False):
    """generates a numpy.linspace of voltage values"""
    return [Volt(step, apply_calibration=apply_calibration)
            for step in numpy.linspace(start, end, steps)]

def linspace_current(start, end, steps, apply_calibration=False):
    """generates a numpy.linspace of current values"""
    return [Ampere(step, apply_calibration=apply_calibration)
            for step in numpy.linspace(start, end, steps)]
