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


voltage_params = [neuron_parameter.E_synx,
                  neuron_parameter.V_synx,
                  neuron_parameter.E_syni,
                  neuron_parameter.V_syni,
                  neuron_parameter.E_l,
                  neuron_parameter.V_t,
                  neuron_parameter.V_exp,
                  neuron_parameter.V_syntci,
                  neuron_parameter.V_syntcx,
                  shared_parameter.V_clra,
                  shared_parameter.V_clrc,
                  shared_parameter.V_reset,
                  shared_parameter.V_dllres,
                  shared_parameter.V_bout,
                  shared_parameter.V_dtc,
                  shared_parameter.V_thigh,
                  shared_parameter.V_br,
                  shared_parameter.V_m,
                  shared_parameter.V_dep,
                  shared_parameter.V_tlow,
                  shared_parameter.V_gmax1,
                  shared_parameter.V_gmax0,
                  shared_parameter.V_gmax3,
                  shared_parameter.V_gmax2,
                  shared_parameter.V_ccas,
                  shared_parameter.V_stdf,
                  shared_parameter.V_fac,
                  shared_parameter.V_bexp,
                  shared_parameter.V_bstdf]


current_params = [neuron_parameter.I_spikeamp,
                  neuron_parameter.I_radapt,
                  neuron_parameter.I_convi,
                  neuron_parameter.I_gl,
                  neuron_parameter.I_convx,
                  neuron_parameter.I_gladapt,
                  neuron_parameter.I_intbbi,
                  neuron_parameter.I_fire,
                  neuron_parameter.I_intbbx,
                  neuron_parameter.I_rexp,
                  neuron_parameter.I_pl,
                  neuron_parameter.I_bexp,
                  shared_parameter.int_op_bias,
                  shared_parameter.I_breset,
                  shared_parameter.I_bstim] 

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
        neuron_parameter.E_syni:    [        1.,  0.],
        neuron_parameter.E_synx:    [        1.,  0.],
        }

def HWtoDAC(value, parameter, rounded = True):
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
            return (value - b)/a
        elif len(HWtoHC_polys[parameter]) is 3:
            a = HWtoHC_polys[parameter][0] 
            b = HWtoHC_polys[parameter][1] 
            c = HWtoHC_polys[parameter][2] 
            return np.sqrt(value/a + b**2 / (4*a**2) - c/a) - b/(2*a)
        else:
            raise ValueError("No valid transformation found.")

def HCtoDAC(value, parameter, rounded = True):
    """ Transform hardware control parameter to DAC value.
        
        Args:
            value = hardware value
            parameter = pyhalbe.HICANN.neuron_parameter or pyhalbe.HICANN.shared_parameter

        Return:
            DAC value
    """
    if parameter in voltage_params:
        DAC = value * 1023/1800.
    elif parameter in current_params:
        DAC = value * 1023/2500.
    
    if rounded:
        DAC = int(round(DAC))

    if (DAC < 1024) and (DAC > -1):
        return DAC
    else:
        if DAC > 1023:
            #print "DAC-Value {} for parameter {} too high. Setting to 1023.".format(DAC,parameter.name)
            return 1023
        if DAC < 0:
            #print "DAC-Value {} for parameter {} too low. Setting to 0.".format(DAC,parameter.name)
            return 0
        #raise ValueError("DAC-Value {} for parameter {} out of range".format(DAC,parameter.name))

def DACtoHC(value, parameter):
    """ Transform DAC value to hardware control parameter.
        
        Args:
            value = DAC value
            parameter = pyhalbe.HICANN.neuron_parameter or pyhalbe.HICANN.shared_parameter

        Return:
            hardware value
    """
    if parameter in voltage_params:
        return value * 1800./1023.
    elif parameter in current_params:
        return value * 2500./1023.

def BiotoHW(value, parameter):
    if parameter in voltage_params:
        if value < 200 and value > -215:
            return 100/20.*value + 900 # dV = 20 mV -> dV = 100 mV. V = -60 mV -> V = 600 mV

def HWtoBio(value, parameter):
    if parameter in voltage_params:
        if value < 1800 and value > 0:
            return 20/100.*value - 180 # dV = 20 mV -> dV = 100 mV. V = -60 mV -> V = 600 mV
