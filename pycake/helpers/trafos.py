""" All important transformations are defined here. Should later be transferred to calibtic.

There are four parameter types:
    Bio: pyNN parameters
    HW: Hardware parameter (what you want to have on the hardware, e.g. V_mem, g_l, tau_ref, ...)
    HC: Hardware control parameter (voltages and currents that are set in the hardware, e.g. I_gl, I_pl instead of g_l and tau_ref...)
    DAC: 10 bit floating gate values. Calibration will be applied to these values

Bio is still work in progress
"""
import numpy as np
from pyhalbe.HICANN import neuron_parameter, shared_parameter
from pyhalbe.HICANN import isCurrentParameter, isVoltageParameter

# Polynomials for the transformation from Hardware to Hardwarecontrol parameters
# Polynomial coefficients are in this order: a*x^2 + b*x + c
# These polynomial coefficients are the results of Marcos transistor-level simulations
HWtoHC_polys = {
        neuron_parameter.E_l:       [        1.,  0.],
        neuron_parameter.V_t:       [        1.,  0.],
        shared_parameter.V_reset:   [        1.,  0.],
        # My measured parameters from /home/np001/experiments/I_gl_ideal_curve/
        neuron_parameter.I_gl:      [  4.90185342e+14,  -1.19763859e+09,   9.38291537e+02],
        #neuron_parameter.I_gl:      [4.32743990e-03, -3.08259218e+00, 9.79811614e+02], # <-- Marco Parameters
        neuron_parameter.I_pl:      [ 0.025e6, 0.0004],
        neuron_parameter.E_syni:    [        1.,  0.],
        neuron_parameter.E_synx:    [        1.,  0.],
        }

def HWtoDAC(value, parameter, rounded=True):
    return HCtoDAC(HWtoHC(value, parameter), parameter, rounded)

def DACtoHW(value, parameter):
    return HCtoHW(DACtoHC(value, parameter), parameter)

def HWtoHC(value, parameter):
    """ Transform hardware parameter to Hardwarecontrol parameter.

        Args:
            value = hardware value
            parameter = pyhalbe.HICANN.neuron_parameter or pyhalbe.HICANN.shared_parameter

        Return:
            Hardwarecontrol value
    """
    if parameter in HWtoHC_polys.keys():
        if parameter is neuron_parameter.I_pl:
            return 1/np.polyval(HWtoHC_polys[parameter], value)
        return np.polyval(HWtoHC_polys[parameter], value)
    else:
        return value

def HCtoHW(value, parameter):
    """ Transform hardware parameter to DAC value.

        Args:
            value = hardware value
            parameter = pyhalbe.HICANN.neuron_parameter or pyhalbe.HICANN.shared_parameter

        Return:
            DAC value
    """
    if parameter in HWtoHC_polys.keys():
        if len(HWtoHC_polys[parameter]) is 2:
            a = HWtoHC_polys[parameter][0]
            b = HWtoHC_polys[parameter][1]
            if parameter is neuron_parameter.I_pl:
                return 1/(a*value) - b/a
            return (value - b)/a
        elif len(HWtoHC_polys[parameter]) is 3:
            a = HWtoHC_polys[parameter][0]
            b = HWtoHC_polys[parameter][1]
            c = HWtoHC_polys[parameter][2]
            return np.sqrt(value/a + b**2 / (4*a**2) - c/a) - b/(2*a)
        else:
            raise ValueError("No valid transformation found.")

def HCtoDAC(value, parameter, rounded=True):
    """ Transform hardware control parameter to DAC value.

        Args:
            value = hardware value
            parameter = pyhalbe.HICANN.neuron_parameter or pyhalbe.HICANN.shared_parameter

        Return:
            DAC value
    """
    lower_limit = 0
    upper_limit = 1023
    if parameter is neuron_parameter.I_pl:
        lower_limit = 4
    if isVoltageParameter(parameter):
        DAC = value * 1023/1.8
    elif isCurrentParameter(parameter):
        DAC = value * 1023/2.5e-6
    else:
        raise ValueError("Invalid Value")

    if rounded:
        DAC = np.round(DAC)
        if np.isscalar(DAC):
            DAC = int(DAC)
        else:
            DAC = np.array(DAC, dtype=int)
    DAC = np.clip(DAC, lower_limit, upper_limit)
    return DAC

def DACtoHC(value, parameter):
    """ Transform DAC value to hardware control parameter.

        Args:
            value = DAC value
            parameter = pyhalbe.HICANN.neuron_parameter or pyhalbe.HICANN.shared_parameter

        Return:
            hardware value
    """
    if isVoltageParameter(parameter):
        return value * 1.8/1023.
    elif isCurrentParameter(parameter):
        return value * 2500/1023.

def BiotoHW(value, parameter):
    if parameter in voltage_params:
        if value < 200 and value > -215:
            return 100/20.*value + 900 # dV = 20 mV -> dV = 100 mV. V = -60 mV -> V = 600 mV

def HWtoBio(value, parameter):
    if parameter in voltage_params:
        if value < 1.8 and value > 0:
            return 20/100.*value - 180 # dV = 20 mV -> dV = 100 mV. V = -60 mV -> V = 600 mV
