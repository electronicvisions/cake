#/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

from pycake.analyzer import Analyzer
from pycake.experimentbuilder import BaseExperimentBuilder
from pycake.helpers.psp_shapes import DoubleExponentialPSP
from pycake.helpers.TraceAverager import createTraceAverager
from pycake.helpers.SignalToNoise import SignalToNoise
from pycake.helpers.sthal import StHALContainer, UpdateParameter
from pycake.measure import Measurement
from pycake.experiment import IncrementalExperiment
from pycake.helpers import psp_fit

import pyhalbe
neuron_parameter = pyhalbe.HICANN.neuron_parameter
from Coordinate import Enum

import numpy
import numpy as np

class V_syntc_PrerunAnalyzer(Analyzer):
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

class PSPAnalyzer(Analyzer):
    """Evaluation class for membrane traces

    This class performs fits on the membrane traces and evaluates
    its quality.
    The class decides by the signal to noise ratio and the chi**2 of the
    fit if it is good enough.
    """
    def __init__(self, averager, bg_period, sn_th, chi2_th, PSPShape,
            prerun, save_mean):
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
        self.save_mean = save_mean

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
        else:
            result['fit'] = None
            result['chi2'] = numpy.nan
            result['err_estimate'] = numpy.nan
            result['area'] = numpy.nan
        if self.save_mean:
            result['mean'] = psp
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

class V_syntc_Generator(object):
    def __init__(self, steps, neurons, readout_shifts):
        self.steps = copy.deepcopy(steps)
        self.neurons = neurons
        self.readout_shifts = readout_shifts

    def __len__(self):
        return len(self.steps)

    def __call__(self, sthal):
        def generator(sthal, steps):
            "Experiment generator"
            for parameters, floating_gates in steps:
                cfg = UpdateParameter(parameters)
                sthal.hicann.floating_gates = floating_gates
                yield cfg, Measurement(sthal, self.neurons, self.readout_shifts)
        return generator(sthal, self.steps)

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

    def get_analyzer(self):
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
        sthal.hicann.floating_gates = self.prepare_parameters(parameters)
        self.set_recording_time(sthal)
        pre_measurement = Measurement(sthal, self.neurons, readout_shifts)
        averager = createTraceAverager(coord_wafer, coord_hicann)

        pre_result = pre_measurement.run_measurement(
                V_syntc_PrerunAnalyzer(averager, self.get_bg_period(sthal)))

        save_traces = self.config.get_save_traces()
        return PSPAnalyzer(averager, self.get_bg_period(sthal),
                self.sn_threshold, self.chi2_threshold,
                DoubleExponentialPSP, pre_result, save_mean=True)
                #save_mean=save_traces)

    def generate_measurements(self):
        self.logger.INFO("Building experiment {}".format(self.config.get_target()))
        coord_wafer, coord_hicann = self.config.get_coordinates()
        steps = self.config.get_steps()

        if not steps:
            raise RuntimeError(":P")

        readout_shifts = self.get_readout_shifts(self.neurons)
        wafer_cfg = self.config.get_wafer_cfg()

        parameters = [[p for p in s.iterkeys()
                       if isinstance(p, neuron_parameter)] for s in steps]
        floating_gates = [self.prepare_parameters(self.get_step_parameters(s))
                          for s in steps]

        sthal = StHALContainer(coord_wafer, coord_hicann, wafer_cfg=wafer_cfg)
        sthal = self.prepare_specific_config(sthal)
        sthal.hicann.floating_gates = copy.copy(floating_gates[0])

        fgcs = [
            sthal.hicann.floating_gates.getFGConfig(Enum(ii)) for ii in (2,3)]
        for fg in floating_gates:
            fg.setNoProgrammingPasses(Enum(len(fgcs)))
            for ii, fgc in enumerate(fgcs):
                fg.setFGConfig(Enum(ii), fgc)


        generator = V_syntc_Generator(zip(parameters, floating_gates),
                                      self.neurons, readout_shifts)
        return sthal, generator

    def get_experiment(self):
        """
        """
        sthal, generator = self.generate_measurements()
        analyzer = self.get_analyzer()
        return IncrementalExperiment(sthal, generator, analyzer)

class V_syntci_Experimentbuilder(V_syntc_Experimentbuilder):
    pass

class V_syntcx_Experimentbuilder(V_syntc_Experimentbuilder):
    pass
