#/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas
import numpy as np
from scipy.optimize import curve_fit

from pyhalbe.HICANN import neuron_parameters
from pycake.analyzer import Analyzer
from pycake.experimentbuilder import BaseExperimentBuilder
from pycake.helpers.peakdetect import peakdet


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

        def cap_voltage(x, tau, offset):
            """charging or discharging of capacitor"""
            # E_l is in mV -> divide by 1000
            final_potential = known_E_l/1.e3
            return (final_potential - offset)*(1 - np.exp(-x/tau)) + offset

        cutout = df[t_min:t_max]

        xData = cutout['t']
        yData = cutout['v']

        # initial values for fit
        guess = [(t_max-t_min), 0.]

        # use inverse-mean of the data set as scale factor
        # to be passed to the underlying leastsq() called by curve_fit()
        diag = (1./xData.mean(), 1./yData.mean())

        fitParams, fitCovariances = curve_fit(cap_voltage, xData, yData, guess, diag=diag)
        return fitParams, fitCovariances

    def __call__(self, neuron, time, voltage, **kwargs):
        """
        Find multiple charging intervals, not using TraceAverager.
        Fit tau_m to each interval, return mean.
        """
        df = pandas.DataFrame({'v': voltage, 't': time}, index=time)
        calibrated_E_l = kwargs['E_l']

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

            fitParams, fitCovariances = self.fit_taum(t_min, t_max, df, calibrated_E_l)
            tau_m = fitParams[0]
            tau_ms.append(tau_m)
        tau_ms = np.array(tau_ms)
        return {
            'tau_m': np.mean(tau_ms),
        }


class I_gl_charging_Experimentbuilder(BaseExperimentBuilder):
    """This experiment probably only works with transistor level
    simulations, but not in hardware due to noise.

    It needs to know the actual (measurable) resting potential,
    which is supplied to the analyzer in a hacky way here.
    When a better data path is available, please use it.
    """
    def get_analyzer(self):
        return I_gl_charging_Analyzer()

    def prepare_parameters(self, parameters):
        # prepare_parameters is called just before make_measurement
        # grab the target E_l information while it is available
        self.E_l = parameters[neuron_parameters.E_l]
        return super(I_gl_charging_Analyzer, self).prepare_parameters(self, parameters)

    def make_measurement(self, sthal, neurons, readout_shifts):
        measurement = super(I_gl_charging_Analyzer, self
                            ).make_measurement(self, sthal, neurons, readout_shifts)
        measurement.initial_data['E_l'] = self.E_l
