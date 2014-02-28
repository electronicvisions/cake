#/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calibration of synapse parameters."""

import os
import time
import numpy as np
import pylogging
from collections import defaultdict
import pyhalbe
import pycalibtic
import pyredman as redman
import pycake.logic.spikes
import scipy.stats as stats
from pycake.helpers.calibtic import create_pycalibtic_polynomial
from pycake.helpers.sthal import StHALContainer, UpdateAnalogOutputConfigurator
from pycake.helpers.units import Current, Voltage, DAC
from pycake.calibration.base import BaseCalibration, BaseTest
from pycake.helpers.TraceAverager import createTraceAverager
from pycake.helpers import psp_fit
from pycake.helpers.psp_shapes import DoubleExponentialPSP

import pysthal

# Import everything needed for saving:
import pickle
import time
import os

# shorter names
Coordinate = pyhalbe.Coordinate
Enum = Coordinate.Enum
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

class Calibrate_E_synx(BaseCalibration):
    target_parameter = neuron_parameter.E_synx

    def init_experiment(self):
        super(Calibrate_E_synx, self).init_experiment()
        self.E_syni_dist = None
        self.E_synx_dist = None
        self.sthal.stimulateNeurons(5.0e6, 4)

    def plot_result_in_trace(self, t, v, neuron, step_id, rep_id):

        return t, [self.all_results[rep_id][neuron]/1000.]*len(t)

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>1000: self.report_bad_trace(t, v, step_id, rep_id, neuron_id)

        return self.correct_for_readout_shift(np.mean(v) * 1000, neuron_id)
    
    def isbroken(self, coeffs):
        if abs(coeffs[0] - 1) > 0.4:     # Broken if slope of the fit is too high or too small
            return True
        else:
            return False

class Calibrate_E_syni(BaseCalibration):
    target_parameter = neuron_parameter.E_syni

    def init_experiment(self):
        super(Calibrate_E_syni, self).init_experiment()
        self.E_syni_dist = None
        self.E_synx_dist = None
        self.sthal.stimulateNeurons(5.0e6, 4)

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>1000: self.report_bad_trace(t, v, step_id, rep_id, neuron_id)

        return self.correct_for_readout_shift(np.mean(v) * 1000, neuron_id)
    
    def isbroken(self, coeffs):
        if abs(coeffs[0] - 1) > 0.4:     # Broken if slope of the fit is too high or too small
            return True
        else:
            return False


class SignalToNoise(object):
    """Utility class to determine the signal to noise ratio of the neuron membrane
    for regular spike input
    """
    def __init__(self, control, adc_freq, bg_freq):
        """
        Args:
            control: np.array Membrane trace of the neuron without synaptic input
            adc_freq: float Sample frequencey of the ADC
            bg_freq: float Frequence of spike input
        """
        self.trace_len = len(control)
        dt = 1.0/adc_freq
        freq = np.fft.fftfreq(self.trace_len, dt)
        peak = np.searchsorted(freq[:len(freq) / 2], bg_freq)
        self.window = np.array( (-2, -1, 0, 1, 2) ) + peak
        spec = np.fft.fft(control)[self.window]
        self.noise = np.sum(np.abs(spec))

    def __call__(self, trace):
        """
        Calculates the signale to noise ratio for the given trace

        Args:
            trace: np.array Membrane trace of the neuron with synaptic input

        Returns:
            float The signal to noise ratio
        """
        assert self.trace_len == len(trace)
        fft_trace = np.abs(np.fft.fft(trace)[self.window])
        return np.sum(fft_trace) / self.noise

class Calibrate_V_syntc_preperation(BaseCalibration):
    """Does meassurements required to calibrate the synaptic time constants

    To calibrate the time constants a ADC Calibration is needed to average the
    neuron traces properly and we ne an baseline meassurement to determine the
    signal to noise ratio for sitmulating spike signal.
    """
    class Process(object):
        def __init__(self, averager, bg_period):
            self.averager = averager
            self.bg_period = bg_period

        def __call__(self, t, v, neuron, *args):
            sn = SignalToNoise(v, self.averager.adc_freq, 1.0/self.bg_period)
            mean, std, n = self.averager.get_average(v, self.bg_period)
            return {
                    "signal_to_noise" : sn,
                    "error_estimate" : np.std(mean),
                    "n" : n }

    def __init__(self, parameter, bg_freq, bg_period, *args, **kwargs):
        self.target_parameter = parameter
        self.bg_freq = bg_freq
        self.bg_period = bg_period
        super(Calibrate_V_syntc_preperation, self).__init__(*args, **kwargs)

    def init_experiment(self):
        self.logger.INFO("Calibrating ADC")
        self.averager = createTraceAverager(
                self.sthal.coord_wafer, self.sthal.coord_hicann)

        super(Calibrate_V_syntc_preperation, self).init_experiment()
        self.description = "V_syntc_preperatory"
        print "recording_time", self.sthal.recording_time

    def get_steps(self):
        return [defaultdict(dict)]

    def get_process_trace_callback(self):
        return self.Process(self.averager, self.bg_period)

    def get_results(self):
        return self.averager, self.all_results[0]

    def process_calibration_results(self, neuron_ids, parameter, linear_fit=False):
        pass

    def store_calibration_results(self, parameter):
        pass


class Calibrate_V_syntc(BaseCalibration):
    """Base class for calibrating the synaptic inputs.

    """
    class Process(object):
        def __init__(self, averager, bg_period, sn_th, chi2_th, PSPShape, prerun):
            self.averager = averager
            self.bg_period = bg_period
            self.sn_th = sn_th
            self.chi2_th = chi2_th
            self.PSPShape = PSPShape
            self.prerun = prerun

        def __call__(self, t, v, neuron, *args):
            signal_to_noise = self.prerun[neuron]["signal_to_noise"](v)
            fit = None
            if signal_to_noise > self.sn_th:
                fit, chi2 = self.fit_psp(t, v, neuron)
            else:
                fit, chi2 = None, None
            return signal_to_noise, fit, chi2

        def fit_psp(self, times, trace, neuron):
            f = self.PSPShape()
            psp, std, n = self.averager.get_average(trace, self.bg_period)
            error_estimate = self.prerun[neuron]["error_estimate"]
            print "fit_psp:", error_estimate
            psp = self.align_psp(psp)
            ok, parameters, _, chi2 = psp_fit.fit(
                    f, times[:len(psp)], psp, error_estimate, maximal_red_chi2=self.chi2_th)
            if ok:
                return dict(zip(f.parameter_names(), parameters)), chi2
            else:
                return None, None

        def align_psp(self, psp):
            assert len(psp) > 1500
            x = np.argmax(psp) - 230
            return np.roll(psp, -x)


    """Base class to calibrate V_syntcx and V_syntci"""
    def init_experiment(self):
        x = 1973.0 # Prime
        self.bg_freq = self.sthal.hicann.pll_freq / x
        self.bg_period = x / self.sthal.hicann.pll_freq
        self.sthal.recording_time = 200.0 * self.bg_period
        self.sn_threshold = float(self.get_config_with_default("sn_threshold", 1.5))
        self.chi2_threshold = float(self.get_config_with_default("chi2_threshold", 25.0))
        self.PSPShape = DoubleExponentialPSP

        prerun = Calibrate_V_syntc_preperation(
                self.target_parameter, self.bg_freq, self.bg_period, self.neuron_ids,
                self.sthal, self.experiment_parameters, self.bg_period)
        prerun.run_experiment()
        self.averager, self.prerun = prerun.get_results()

        super(Calibrate_V_syntc, self).init_experiment()
        assert (self.sthal.recording_time == 200.0 * self.bg_period)
        self.sthal.stimulateNeurons(self.bg_freq, 1)
        self.save_prerun()

    def save_prerun(self):
        if self.save_results:
            self.pickle(self.averager, self.folder, 'Averager.p')
            self.pickle(self.prerun, self.folder, 'V_syntc_prerun.p')

    def get_process_trace_callback(self):
        return self.Process(self.averager, self.bg_period, self.sn_threshold,
                self.chi2_threshold, self.PSPShape, self.prerun)

    def process_calibration_results(self, neurons, parameter, dim):
        pass

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass

    def isbroken(self, coeffs):
        broken = np.any(np.isnan(coeffs))
        self.logger.ERROR("Isbroken?: {}: {} {}".format(coeffs, broken, type(coeffs)))
        return broken

    def align_psp(self, psp):
        """shifts the averaged psp that it can be fitted easily"""
        raise NotImplemented()

class Calibrate_V_syntcx(Calibrate_V_syntc):
    target_parameter = neuron_parameter.V_syntcx

class Calibrate_V_syntci(Calibrate_V_syntc):
    target_parameter = neuron_parameter.V_syntci

    def align_psp(self, psp):
        assert len(psp) > 1500
        x = np.argmin(psp) - 230
        return np.roll(psp, -x)



class Test_E_synx(BaseTest):
    target_parameter = neuron_parameter.E_synx

    def init_experiment(self):
        super(Test_E_synx, self).init_experiment()
        self.E_syni_dist = None
        self.E_synx_dist = None
        self.sthal.stimulateNeurons(5.0e6, 4)

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>1000: self.report_bad_trace(t, v, step_id, rep_id, neuron_id)

        return self.correct_for_readout_shift(np.mean(v) * 1000, neuron_id)
    
class Test_E_syni(BaseTest):
    target_parameter = neuron_parameter.E_syni

    def init_experiment(self):
        super(Test_E_syni, self).init_experiment()
        self.E_syni_dist = None
        self.E_synx_dist = None
        self.sthal.stimulateNeurons(5.0e6, 4)

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>1000: self.report_bad_trace(t, v, step_id, rep_id, neuron_id)

        return self.correct_for_readout_shift(np.mean(v) * 1000, neuron_id)
