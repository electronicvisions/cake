#/usr/bin/env python
# -*- coding: utf-8 -*-

from pycake.analyzer import Analyzer
from pycake.experimentbuilder import BaseExperimentBuilder
from pycake.helpers.psp_shapes import DoubleExponentialPSP
from pycake.helpers.TraceAverager import createTraceAverager
from pycake.helpers.SignalToNoise import SignalToNoise
from pycake.helpers.sthal import StHALContainer
from pycake.measure import Measurement
from pycake.helpers import psp_fit

import numpy as np

class V_syntc_PrerunAnalyzer(object):
    def __init__(self, averager, bg_period):
        self.averager = averager
        self.bg_period = bg_period

    def __call__(self, t, v, neuron):
        # FFT seems much (10-100x) faster on arrays for size n**2
        sn = SignalToNoise(v, self.averager.adc_freq, 1.0/self.bg_period)
        mean, std, n = self.averager.get_average(v, self.bg_period)
        return {
                "signal_to_noise" : sn,
                "error_estimate" : np.std(mean),
                "error_estimate_samples" : n }

class PSPAnalyzer(object):
    """Evaluation class for membrane traces

    This class performs fits on the membrane traces and evaluates
    its quality.
    The class decides by the signal to noise ratio and the chi**2 of the
    fit if it is good enough.
    """
    def __init__(self, averager, bg_period, sn_th, chi2_th, PSPShape, prerun):
        """
        Args:
            averager: TraceAverager class that can perform the averaging of the
                      membrane trace
            bg_period: float Frequency of spike input
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

class V_syntc_Experimentbuilder(BaseExperimentBuilder):
    def __init__(self, *args, **kwargs):
        super(V_syntc_Experimentbuilder, self).__init__(*args, **kwargs)

        self.samples = 400
        self.sn_threshold = float(
                self.config.get_config_with_default("sn_threshold", 1.5))
        self.chi2_threshold = float(
                self.config.get_config_with_default("chi2_threshold", 25.0))

        self.psp_length = 1973 # Prime

    def get_bg_freq(self, sthal):
        return float(sthal.getPLL()) / self.psp_length

    def get_bg_period(self, sthal):
        return self.psp_length / float(sthal.getPLL())

    def set_recording_time(self, sthal):
        sthal.recording_time = self.samples * self.get_bg_period(sthal)

    def prepare_specific_config(self, sthal):
        sthal.stimulateNeurons(self.get_bg_freq(sthal), 1)
        self.set_recording_time(sthal)
        return sthal

    def get_analyzer(self, parameter):
        """Creates the analyzer for PSP measurements.
        This makes to preparatory measurements.
        1) Initializing the Trace Average
        2) Determine the background noise, for error estimation and signal 
            strength
        """
        coord_wafer, coord_hicann = self.config.get_coordinates()
        parameters = self.config.get_parameters()
        readout_shifts = self.get_readout_shifts(self.neurons)

        sthal = StHALContainer(coord_wafer, coord_hicann)
        sthal = self.prepare_parameters(sthal, parameters)
        self.set_recording_time(sthal)
        pre_measurement = Measurement(sthal, self.neurons, readout_shifts)
        averager = createTraceAverager(coord_wafer, coord_hicann)

        pre_result = pre_measurement.run_measurement(
                V_syntc_PrerunAnalyzer(averager, self.get_bg_period(sthal)))

        import pprint
        pprint.pprint(pre_result)

        return PSPAnalyzer(averager, self.get_bg_period(sthal),
                self.sn_threshold, self.chi2_threshold,
                DoubleExponentialPSP, pre_result)

class V_syntci_Experimentbuilder(V_syntc_Experimentbuilder):
    pass

class V_syntcx_Experimentbuilder(V_syntc_Experimentbuilder):
    pass
