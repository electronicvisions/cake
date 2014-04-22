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
from pycake.helpers.SignalToNoise import SignalToNoise 
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

    def process_trace(self, t, v, neuron, step_id, rep_id):
        if np.std(v)*1000>1000: self.report_bad_trace(t, v, step_id, rep_id, neuron)

        return self.correct_for_readout_shift(np.mean(v) * 1000, neuron)

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

    def process_trace(self, t, v, neuron, step_id, rep_id):
        if np.std(v)*1000>1000: self.report_bad_trace(t, v, step_id, rep_id, neuron)

        return self.correct_for_readout_shift(np.mean(v) * 1000, neuron)

    def isbroken(self, coeffs):
        if abs(coeffs[0] - 1) > 0.4:     # Broken if slope of the fit is too high or too small
            return True
        else:
            return False


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
            # FFT seems much (10-100x) faster on arrays for size n**2
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
        self.repetitions = 1

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

    def process_calibration_results(self, neurons, parameter, linear_fit=False):
        pass

    def store_calibration_results(self, parameter):
        pass


class Calibrate_V_syntc(BaseCalibration):
    """Base class for calibrating the synaptic inputs.

    See subclass Process for detail.
    """
    class Process(object):
        """Evaluation class for membrane traces

        This class performs fits on the membrane traces and evaluates
        its quality.
        The class decides by the signal to noise ratio and the chi**2 of the
        fit if it is good enough.
        """
        def __init__(self, averager, bg_period, sn_th, chi2_th, PSPShape, prerun):
            """
            Args:
                averager: TraceAverager class that can perfome the averaging of the
                          membrane trace
                bg_period: float Frequence of spike input
                sn_th: Minimum signal to noise ratio to accept a trace
                chi2_th: Minimum chi**2 to accept the fit
                PSPShape: class modeling the shape of the psp to fit
                prerun: results of the Calibrate_V_syntc_preperation run
            """
            self.averager = averager
            self.bg_period = bg_period
            self.sn_th = sn_th
            self.chi2_th = chi2_th
            self.PSPShape = PSPShape
            self.prerun = prerun

        def __call__(self, t, v, neuron, *args):
            """
            Performs a fit on a membrane trace

            Args:
                t: np.array Time points of the trace
                v: np.array Membrane trace of the neuron with synaptic input
                neuron: np.array Neuron from which this trace was taken
                *args: ignored

            Returns:
                (float, float, float) containing the signal to noise ratio,
                    the fit parameters and the chi**2 of the fit. The last
                    two values may be None if the fit fails
            """
            signal_to_noise = self.prerun[neuron]["signal_to_noise"](v)
            result = {'signal_to_noise' : signal_to_noise, 'fit' : None}
            if signal_to_noise > self.sn_th:
                result.update(self.fit_psp(t, v, neuron))
            return result

        def fit_psp(self, times, trace, neuron):
            """
            Do the fitting of the membrane trace

            Args:
                times: np.array Time points of the trace
                trace: np.array Membrane trace of the neuron with synaptic input
                neuron: np.array Neuron from which this trace was taken

            Returns:
                (float, float) the fit parameters and the chi**2 of the fit. The
                    two values may be None if the fit fails
            """
            result = {}
            f = self.PSPShape()
            psp, std, n = self.averager.get_average(trace, self.bg_period)
            error_estimate = self.prerun[neuron]["error_estimate"]
            psp = self.align_psp(psp)
            jacobian = getattr(f, "jacobian", None)
            ok, parameters, err_estimate, chi2 = psp_fit.fit(
                    f, times[:len(psp)], psp, error_estimate,
                    maximal_red_chi2=self.chi2_th,
                    jacobian=jacobian)
            if ok:
                result['fit'] = dict(zip(f.parameter_names(), parameters))
                result['chi2'] = chi2
                result['err_estimate'] = err_estimate
                result['area'] = np.sum(psp - result['fit']['offset'])
            return result

        def align_psp(self, psp):
            """Shift the psp that the maximum lays at a certain value
            Args:
                trace: np.array The trace

            Returns:
                np.array shifted trace
            """
            assert len(psp) > 1500
            x = np.argmax(psp) - 230
            return np.roll(psp, -x)


    def init_experiment(self):
        x = 1973.0 # Prime
        PLL = self.stahl.wafer.commonFPGASettings().getPLL()
        self.bg_freq = PLL / x
        self.bg_period = x / PLL
        self.sthal.recording_time = 400.0 * self.bg_period
        self.sn_threshold = float(self.get_config_with_default("sn_threshold", 1.5))
        self.chi2_threshold = float(self.get_config_with_default("chi2_threshold", 25.0))
        self.PSPShape = DoubleExponentialPSP

        prerun = Calibrate_V_syntc_preperation(
                self.target_parameter, self.bg_freq, self.bg_period, self.neurons,
                self.sthal, self.experiment_parameters, self.bg_period)
        prerun.run_experiment()
        self.averager, self.prerun = prerun.get_results()

        super(Calibrate_V_syntc, self).init_experiment()
        assert (self.sthal.recording_time == 400.0 * self.bg_period)
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

    def process_results(self, neurons):
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

    def process_trace(self, t, v, neuron, step_id, rep_id):
        if np.std(v)*1000>1000: self.report_bad_trace(t, v, step_id, rep_id, neuron)

        return self.correct_for_readout_shift(np.mean(v) * 1000, neuron)

class Test_E_syni(BaseTest):
    target_parameter = neuron_parameter.E_syni

    def init_experiment(self):
        super(Test_E_syni, self).init_experiment()
        self.E_syni_dist = None
        self.E_synx_dist = None
        self.sthal.stimulateNeurons(5.0e6, 4)

    def process_trace(self, t, v, neuron, step_id, rep_id):
        if np.std(v)*1000>1000: self.report_bad_trace(t, v, step_id, rep_id, neuron)

        return self.correct_for_readout_shift(np.mean(v) * 1000, neuron)
