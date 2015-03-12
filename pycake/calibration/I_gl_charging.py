#/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import numpy as np
from scipy.optimize import curve_fit

from pyhalbe.HICANN import neuron_parameter
from pycake.analyzer import Analyzer
from pycake.measure import ADCMeasurement
from pycake.experimentbuilder import BaseExperimentBuilder
from pycake.helpers.peakdetect import peakdet
from pycake.helpers.units import Volt


class I_gl_charging_Analyzer(Analyzer):
    """Fit exponential curve when charging the capacitor.

    Assumes that E_l > V_t, but only slightly larger.
    Adaptation is supposed to be disabled.
    """

    def fit_taum(self, df, t_min, t_max, known_E_l):
        """Fit exponential charge from t_min to t_max.

        Args:
            df: DataFrame containing keys 't','v', index t
                time in [s], voltage in [V]
            known_E_l [mV]: potential which is approached by charging
        """

        def cap_voltage(t, tau, offset):
            """charging or discharging of capacitor"""
            # E_l is in mV -> divide by 1000
            final_potential = known_E_l/1.e3
            return offset * np.exp(-t/tau) + final_potential

        cutout = df[t_min:t_max]

        xData = cutout['t']
        yData = cutout['v']

        interval_x = t_max - t_min
        interval_y = np.max(yData) - np.min(yData)

        # initial values for fit
        guess = [interval_x, np.min(yData)]

        # use inverse-interval of the data set as scale factor
        # to be passed to the underlying leastsq() called by curve_fit()
        rescale = (1./interval_x, 1./interval_y)

        fitParams, fitCovariances = curve_fit(cap_voltage, xData - t_min, yData, guess, diag=rescale)

        return fitParams, fitCovariances

    def __call__(self, neuron, t, v, **kwargs):
        """
        Find multiple charging intervals, not using TraceAverager.
        Fit tau_m to each interval, return mean.
        """
        df = pandas.DataFrame({'v': v, 't': t}, index=t)

        # get E_l via hacky data path
        calibrated_E_l = kwargs['E_l'][0]

        maxtab, mintab = peakdet(df['v'].values, 0.03, df['t'].values)

        # iterate over detected intervals
        tau_ms = []
        num_max = len(maxtab)
        for idxmin in range(len(mintab)):
            t_min = mintab[idxmin][0]
            if maxtab[idxmin][0] > t_min:
                t_max = maxtab[idxmin][0]
            else:
                # use next maximum
                if idxmin + 1 == num_max:
                    # no further maximum in trace, stop
                    break
                t_max = maxtab[idxmin + 1][0]

            # add buffer around detected edges
            t_buffer = 1e-8
            t_min = t_min + t_buffer
            t_max = t_max - t_buffer

            fitParams, fitCovariances = self.fit_taum(df, t_min, t_max, calibrated_E_l)
            tau_m = fitParams[0]
            tau_ms.append(tau_m)
            # TODO check quality of fit by calculating difference to actual
            # trace, return error estimation
        tau_ms = np.array(tau_ms)
        return {
            'tau_m': np.mean(tau_ms),
        }


class AddElADCMeasurement(ADCMeasurement):
    """This is a hack to add E_l information to the analyzer"""

    def __init__(self, sthal, neurons, readout_shifts, E_l):
        super(AddElADCMeasurement, self).__init__(sthal, neurons, readout_shifts)
        self.E_l = E_l

    def read_adc(self):
        readout = super(AddElADCMeasurement, self).read_adc()
        if readout is not None:
            readout['E_l'] = np.array([self.E_l])
        return readout


class I_gl_charging_Experimentbuilder(BaseExperimentBuilder):
    """This experiment probably only works with transistor level
    simulations, but not in hardware due to noise.

    It needs to know the actual (measurable) resting potential,
    which is supplied to the analyzer in a hacky way here.
    When a better data path is available, please use it.
    """
    def get_analyzer(self):
        return I_gl_charging_Analyzer()

    def prepare_specific_config(self, sthal):
        sthal.maximum_spikes = 10
        return sthal

    def prepare_parameters(self, parameters):
        # prepare_parameters is called just before make_measurement
        # grab the target E_l information while it is available
        E_l = parameters[neuron_parameter.E_l]
        if type(E_l) is not Volt:
            raise TypeError("E_l must be Volt()")
        self.E_l = E_l.value  # in mV
        return super(I_gl_charging_Experimentbuilder, self).prepare_parameters(parameters)

    def make_measurement(self, sthal, neurons, readout_shifts):
        """This is a hack to add E_l information to the analyzer"""
        return AddElADCMeasurement(sthal, neurons, readout_shifts, self.E_l)
