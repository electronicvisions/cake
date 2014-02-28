#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calibration of LIF parameters."""

import numpy as np
import pylab
import scipy
from scipy.optimize import curve_fit
import pylogging
from collections import defaultdict
import pyhalbe
import pycalibtic
import pyredman as redman
import pycake.logic.spikes
from pycake.helpers.calibtic import create_pycalibtic_polynomial
from pycake.helpers.sthal import StHALContainer, UpdateAnalogOutputConfigurator
from pycake.helpers.units import Current, Voltage, DAC
from pycake.helpers.trafos import HWtoDAC, HCtoDAC, DACtoHC, DACtoHW
from pycake.calibration.base import BaseCalibration, BaseTest
from pycake.helpers.TraceAverager import createTraceAverager
import pycake.helpers.misc as misc

# Import everything needed for saving:
import pickle
import time
import os

# shorter names
Coordinate = pyhalbe.Coordinate
Enum = Coordinate.Enum
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

def find_baseline(t,v):
        """ find baseline of trace

            t - list of time stamps
            v - corresponding values

            returns baseline and time between values in unit index (do t[delta_t] to get delta in units of time)
        """

        std_v = np.std(v)

        # to be tuned
        drop_threshold = -std_v/2
        min_drop_distance = 5
        start_div = 10
        end_div = 5

        # find right edge of spike --------------------------------------------
        diff = np.diff(v)
        drop = np.where(diff < drop_threshold)[0]
        #----------------------------------------------------------------------

        # the differences of the drop position n yields the time between spikes
        drop_diff = np.diff(drop)
        # filter consecutive values
        drop_diff_filtered = drop_diff[drop_diff > min_drop_distance]
        # take mean of differences
        delta_t = np.mean(drop_diff_filtered)
        #----------------------------------------------------------------------

        # collect baseline voltages -------------------------------------------
        only_base = []
        last_n = -1

        baseline = 0
        N = 0

        for n in drop[np.where(np.diff(drop) > min_drop_distance)[0]]:

            # start some time after spike and stop early to avoid taking rising edge
            start = n+int(delta_t/start_div)
            end = n+int(delta_t/end_div)

            if start < len(v) and end < len(v):
                baseline += np.mean(v[start:end])
                N += 1

        # take mean
        if N:
            baseline /= N
        else:
            baseline = np.min(v)

        #----------------------------------------------------------------------

        return baseline, delta_t

class Calibrate_E_l(BaseCalibration):
    """E_l calibration."""
    target_parameter = neuron_parameter.E_l

    def plot_result_in_trace(self, t, v, neuron, step_id, rep_id):

        return t, [self.all_results[rep_id][neuron]/1000.]*len(t)

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>50: self.report_bad_trace(t, v, step_id, rep_id, neuron_id)
        #return np.mean(v)*1000 # Get the mean value * 1000 for mV
        return self.correct_for_readout_shift(np.mean(v)*1000, neuron_id) # Get the mean value * 1000 for mV


class Calibrate_V_t(BaseCalibration):
    """V_t calibration."""
    target_parameter = neuron_parameter.V_t

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000<5: self.report_bad_trace(t, v, step_id, rep_id, neuron_id)
        # Return max value. This should be more accurate than a mean value of all maxima because the ADC does not always hit the real maximum value, underestimating V_t.
        return self.correct_for_readout_shift(np.max(v)*1000, neuron_id)


class Calibrate_V_reset(BaseCalibration):
    """V_reset calibration."""
    target_parameter = shared_parameter.V_reset

    def plot_result_in_trace(self, t, v, neuron, step_id, rep_id):

        return t, [self.all_results[rep_id][neuron]/1000.]*len(t)

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000<5: self.report_bad_trace(t, v, step_id, rep_id, neuron_id)

        baseline, delta_t = find_baseline(t,v)

        return baseline * 1000


    def process_calibration_results(self, neurons, parameter, dim):
        """This base class function can be used by child classes as process_results."""
        # containers for final results
        self.results_mean = defaultdict(list)
        self.results_std = defaultdict(list)
        self.results_polynomial = {}

        # self.all_results contains all measurements, need to untangle
        # repetitions and steps
        # This is done for shared and neuron parameters
        repetition = 1
        step_results = defaultdict(list)
        for result in self.all_results:
            # still in the same step, collect repetitions for averaging
            for neuron in neurons:
                step_results[neuron].append(result[neuron])
            repetition += 1
            if repetition > self.repetitions:
                # step is done; average, store and reset
                for neuron in neurons:
                    mean = HWtoDAC(np.mean(step_results[neuron]), parameter)
                    std = HWtoDAC(np.std(step_results[neuron]), parameter)
                    self.results_mean[neuron].append(mean)
                    self.results_std[neuron].append(std)
                repetition = 1
                step_results = defaultdict(list)

        # For shared parameters: mean over one block
        def iter_v_reset(block):
            neuron_on_quad = Coordinate.NeuronOnQuad(block.x(), block.y())
            for quad in Coordinate.iter_all(Coordinate.QuadOnHICANN):
                yield Coordinate.NeuronOnHICANN(quad, neuron_on_quad)
            return

        self.results_mean_shared = defaultdict(list)
        self.results_std_shared = defaultdict(list)
        for block in self.get_blocks():
            neurons_on_block = [n for n in iter_v_reset(block)]
            self.results_mean_shared[block] = np.mean([self.results_mean[n_coord] for n_coord in neurons_on_block], axis = 0)
            self.results_std_shared[block] = np.mean([self.results_std[n_coord] for n_coord in neurons_on_block], axis = 0)

        steps = self.get_steps()

        self.results_diff_mean = defaultdict(list)
        self.results_diff_std = defaultdict(list)

        for neuron in self.get_neurons():
            block = neuron.sharedFGBlock()
            self.results_diff_mean[neuron] = [self.results_mean[neuron][step_id] - self.results_mean_shared[block][step_id] for step_id in range(len(steps))]
            self.results_diff_std[neuron] = [np.sqrt(self.results_std[neuron][step_id]**2 + self.results_std_shared[block][step_id]**2) for step_id in range(len(steps))] 

        for neuron in self.get_neurons():
            block = neuron.sharedFGBlock()
            self.results_polynomial[neuron] = self.do_fit(coord=block,
                                                          parameter=shared_parameter.V_reset,
                                                          steps=steps,
                                                          mean=self.results_diff_mean[neuron],
                                                          std=self.results_diff_std[neuron],
                                                          dim=0,
                                                          swap_fit_x_y=True)

        for block in self.get_blocks():
            self.results_polynomial[block] = self.do_fit(coord=block,
                                                         parameter=shared_parameter.V_reset,
                                                         steps=steps,
                                                         mean=self.results_mean_shared[block],
                                                         std=self.results_std_shared[block],
                                                         dim=1,
                                                         swap_fit_x_y=False)

    def store_results(self):
        # Store readout shift as 21st parameter
        self.store_calibration_results(21, isneuron=True)
        self.store_calibration_results(shared_parameter.V_reset)


class Calibrate_I_gl(BaseCalibration):
    target_parameter = neuron_parameter.I_gl

    def init_experiment(self):
        super(Calibrate_I_gl, self).init_experiment()
        self.sthal.recording_time = 5e-3
        self.stim_length = 65
        self.pulse_length = 15
        self.stim_current = 35          # Stim current in nA

        # Get the trace averager
        self.logger.INFO("{}: Creating trace averager".format(time.asctime()))
        coord_hglobal = self.sthal.hicann.index()  # grab HICANNGlobal from StHAL
        coord_wafer = coord_hglobal.wafer()
        coord_hicann = coord_hglobal.on_wafer()
        self.trace_averager = createTraceAverager(coord_wafer, coord_hicann)
        self.logger.INFO("{}: Trace averager created with ACD clock of {} Hz".format(time.asctime(), self.trace_averager.adc_freq))

    def prepare_measurement(self, step_parameters, step_id, rep_id):
        """Prepare measurement. This is done for each repetition,
        Perform reset, write general hardware settings.
        For I_gl measurement: Set current stimulus
        """
        stimulus = pyhalbe.HICANN.FGStimulus()
        stimulus.setPulselength(self.pulse_length)
        stimulus.setContinuous(True)

        stimulus[:self.stim_length] = [self.stim_current] * self.stim_length
        stimulus[self.stim_length:] = [0] * (len(stimulus) - self.stim_length)

        self.sthal.set_current_stimulus(stimulus)

        if self.bigcap is True:
            # TODO add bigcap functionality
            pass

        if self.save_results:
            fgc = self.sthal.hicann.floating_gates
            misc.mkdir_p(os.path.join(self.folder,"floating_gates"))
            pickle.dump(fgc, open(os.path.join(self.folder,"floating_gates", "step{}rep{}.p".format(step_id,rep_id)), 'wb'))

        self.sthal.write_config()


    def measure(self, neuron_ids, step_id, rep_id):
        """Perform measurements for a single step on one or multiple neurons."""
        results = {}
        chisquares = {}
        # This magic number is the temporal distance of one current pulse repetition to the next one:
        dt = 129 * 4 * (self.pulse_length + 1) / self.sthal.hicann.pll_freq
        for neuron_id in neuron_ids:
            self.sthal.switch_current_stimulus(neuron_id)
            t, v = self.sthal.read_adc()

            # Convert the whole trace into a mean trace
            mean_trace, std_trace, n_mean = self.trace_averager.get_average(v, dt)
            mean_trace = np.array(mean_trace)
            std_trace = np.array(std_trace)
            std_trace /= np.sqrt(np.floor(len(v)/len(mean_trace)))

            t = t[0:len(mean_trace)]
            v = mean_trace

            # Save traces in files:
            if self.save_traces:
                self.save_trace(t, v, neuron_id, step_id, rep_id)

            results[neuron_id], chisquares[neuron_id] = self.process_trace(t, mean_trace, std_trace, neuron_id, step_id, rep_id)

        # Now store measurements in a file:
        if self.save_results:
            self.save_result(results, step_id, rep_id)
            #if not os.path.isdir(os.path.join(self.folder,"chisquares/")):
            #    os.mkdir(os.path.join(self.folder,"chisquares/"))
            #pickle.dump(chisquares, open(os.path.join(self.folder,"chisquares/","step{}_rep{}.p".format(step_id, rep_id)), 'wb'))
        self.all_results.append(results)


    def process_trace(self, t, mean_trace, std_trace, neuron_id, step_id, rep_id):
        # Capacity if bigcap is turned on:
        C = 2.16456e-12
        tau_m, red_chisquare = self.fit_exponential(mean_trace, std_trace)
        g_l = C / tau_m
        return g_l, red_chisquare

    def get_decay_fit_range(self, trace):
        """Cuts the trace for the exponential fit. This is done by calculating the second derivative."""
        filter_width = 250e-9 # Not a magic number! This was carefully tuned to give best results
        dt = 1/self.trace_averager.adc_freq
    
        diff = pylab.roll(trace, 1) - trace
        diff2 = diff - pylab.roll(diff, -1)
        diff2 /= (dt ** 2)
    
        kernel = scipy.signal.gaussian(len(trace), filter_width / dt)
        kernel /= pylab.sum(kernel)
        kernel = pylab.roll(kernel, int(len(trace) / 2))
        diff2_smooth = pylab.real(pylab.ifft(pylab.fft(kernel) * pylab.fft(diff2)))

        fitstart = np.argmin(diff2_smooth)
        fitstop = np.argmax(diff2_smooth)

        if fitstart > fitstop:
            trace = pylab.roll(trace, -fitstart)
            trace_cut = trace[:fitstop-fitstart+len(trace)]
            fittime = np.arange(0, fitstop-fitstart+len(trace))
        else:
            trace_cut = trace[fitstart:fitstop]
            fittime = np.arange(fitstart, fitstop)

        return trace_cut, fittime
    
    def fit_exponential(self, mean_trace, std_trace, stim_length = 65):
        """ Fit an exponential function to the mean trace. """
        func = lambda x, tau, offset, a: a * np.exp(-(x - x[0]) / tau) + offset
    
        trace_cut, fittime = self.get_decay_fit_range(mean_trace)
    
        expf, pcov, infodict, errmsg, ier = curve_fit(
            func,
            fittime,
            trace_cut,
            [.5, 100., 0.1],
            full_output=True)
            #sigma=std_trace[fittime],
    
        tau = expf[0] / self.trace_averager.adc_freq 

        DOF = len(fittime) - len(expf)
        red_chisquare = sum(infodict["fvec"] ** 2) / (DOF)
    
        return tau, red_chisquare

    def process_results(self, neuron_ids):
        # I_gl should be fitted with 2nd degree polynomial
        self.process_calibration_results(neuron_ids, self.target_parameter, 2)



"""
        Measurement classes start here. These classes are used to test if the calibration was successful.
        They do this by measuring with every calibration turned on without processing results.
"""

class Test_E_l(BaseTest):
    """E_l calibration."""
    target_parameter = neuron_parameter.E_l

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>50: self.report_bad_trace(t, v, step_id, rep_id, neuron_id)

        #return np.mean(v)*1000 # Get the mean value * 1000 for mV
        return self.correct_for_readout_shift(np.mean(v)*1000, neuron_id) # Get the mean value * 1000 for mV


class Test_V_t(BaseTest):
    """V_t calibration."""
    target_parameter = neuron_parameter.V_t

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000<5: self.report_bad_trace(t, v, step_id, rep_id, neuron_id)

        # Return max value. This should be more accurate than a mean value of all maxima because the ADC does not always hit the real maximum value, underestimating V_t.
        return self.correct_for_readout_shift(np.max(v)*1000, neuron_id)


class Test_V_reset(BaseTest):
    """V_reset calibration."""
    target_parameter = shared_parameter.V_reset

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000<5: self.report_bad_trace(t, v, step_id, rep_id, neuron_id)

        baseline, delta_t = find_baseline(t,v)

        return self.correct_for_readout_shift(baseline * 1000, neuron_id)

class Test_I_gl(BaseTest):
    target_parameter = neuron_parameter.I_gl

    def init_experiment(self):
        super(Test_I_gl, self).init_experiment()
        self.sthal.recording_time = 5e-3
        self.stim_length = 65
        self.pulse_length = 15

        # Get the trace averager
        self.logger.INFO("{}: Creating trace averager".format(time.asctime()))
        coord_hglobal = self.sthal.hicann.index()  # grab HICANNGlobal from StHAL
        coord_wafer = coord_hglobal.wafer()
        coord_hicann = coord_hglobal.on_wafer()
        self.trace_averager = createTraceAverager(coord_wafer, coord_hicann)
        self.logger.INFO("{}: Trace averager created with ACD clock of {} Hz".format(time.asctime(), self.trace_averager.adc_freq))

        # TEMPORARY SOLUTION:
        # self.save_traces = True

    def prepare_measurement(self, step_parameters, step_id, rep_id):
        """Prepare measurement. This is done for each repetition,
        Perform reset, write general hardware settings.
        For I_gl measurement: Set current stimulus
        """
        stimulus = pyhalbe.HICANN.FGStimulus()
        stimulus.setPulselength(self.pulse_length)
        stimulus.setContinuous(True)

        stimulus[:self.stim_length] = [35] * self.stim_length
        stimulus[self.stim_length:] = [0] * (len(stimulus) - self.stim_length)

        self.sthal.set_current_stimulus(stimulus)

        if self.bigcap is True:
            # TODO add bigcap functionality
            pass

        if self.save_results:
            fgc = self.sthal.hicann.floating_gates
            misc.mkdir_p(os.path.join(self.folder,"floating_gates"))
            pickle.dump(fgc, open(os.path.join(self.folder,"floating_gates", "step{}rep{}.p".format(step_id,rep_id)), 'wb'))

        self.sthal.write_config()


    def measure(self, neuron_ids, step_id, rep_id):
        """Perform measurements for a single step on one or multiple neurons."""
        results = {}
        chisquares = {}
        # This magic number is the temporal distance of one current pulse repetition to the next one:
        dt = 129 * 4 * (self.pulse_length + 1) / self.sthal.hicann.pll_freq
        for neuron_id in neuron_ids:
            self.sthal.switch_current_stimulus(neuron_id)
            t, v = self.sthal.read_adc()

            # Convert the whole trace into a mean trace
            mean_trace, std_trace, n_mean = self.trace_averager.get_average(v, dt)
            mean_trace = np.array(mean_trace)
            std_trace = np.array(std_trace)
            std_trace /= np.sqrt(np.floor(len(v)/len(mean_trace)))

            t = t[0:len(mean_trace)]
            v = mean_trace

            # Save traces in files:
            if self.save_traces:
                self.save_trace(t, v, neuron_id, step_id, rep_id)

            results[neuron_id], chisquares[neuron_id] = self.process_trace(t, mean_trace, std_trace, neuron_id, step_id, rep_id)

        # Now store measurements in a file:
        if self.save_results:
            self.save_result(results, step_id, rep_id)
            #if not os.path.isdir(os.path.join(self.folder,"chisquares/")):
            #    os.mkdir(os.path.join(self.folder,"chisquares/"))
            #pickle.dump(chisquares, open(os.path.join(self.folder,"chisquares/","step{}_rep{}.p".format(step_id, rep_id)), 'wb'))
        self.all_results.append(results)


    def process_trace(self, t, mean_trace, std_trace, neuron_id, step_id, rep_id):
        # Capacity if bigcap is turned on:
        C = 2.16456e-12
        tau_m, red_chisquare = self.fit_exponential(mean_trace, std_trace)
        g_l = C / tau_m
        return g_l, red_chisquare

    def get_decay_fit_range(self, trace):
        """Cuts the trace for the exponential fit. This is done by calculating the second derivative."""
        filter_width = 250e-9 # Not a magic number! This was carefully tuned to give best results
        dt = 1/self.trace_averager.adc_freq
    
        diff = pylab.roll(trace, 1) - trace
        diff2 = diff - pylab.roll(diff, -1)
        diff2 /= (dt ** 2)
    
        kernel = scipy.signal.gaussian(len(trace), filter_width / dt)
        kernel /= pylab.sum(kernel)
        kernel = pylab.roll(kernel, int(len(trace) / 2))
        diff2_smooth = pylab.real(pylab.ifft(pylab.fft(kernel) * pylab.fft(diff2)))

        fitstart = np.argmin(diff2_smooth)
        fitstop = np.argmax(diff2_smooth)

        if fitstart > fitstop:
            trace = pylab.roll(trace, -fitstart)
            trace_cut = trace[:fitstop-fitstart+len(trace)]
            fittime = np.arange(0, fitstop-fitstart+len(trace))
        else:
            trace_cut = trace[fitstart:fitstop]
            fittime = np.arange(fitstart, fitstop)

        return trace_cut, fittime
    
    def fit_exponential(self, mean_trace, std_trace, stim_length = 65):
        """ Fit an exponential function to the mean trace. """
        func = lambda x, tau, offset, a: a * np.exp(-(x - x[0]) / tau) + offset
    
        trace_cut, fittime = self.get_decay_fit_range(mean_trace)
    
        expf, pcov, infodict, errmsg, ier = curve_fit(
            func,
            fittime,
            trace_cut,
            [.5, 100., 0.1],
            full_output=True)
            #sigma=std_trace[fittime],
    
        tau = expf[0] / self.trace_averager.adc_freq 

        DOF = len(fittime) - len(expf)
        red_chisquare = sum(infodict["fvec"] ** 2) / (DOF)
    
        return tau, red_chisquare

    pass




"""
        EVERYTHING AFTER THIS IS POINT STILL UNDER CONSTRUCTION
"""



# TODO
class Calibrate_tau_ref(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_tau_ref, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                neuron_parameter.E_l: Voltage(800),
                neuron_parameter.V_t: Voltage(700),
                shared_parameter.V_reset: Voltage(500),
                neuron_parameter.I_gl: Current(1000),
                neuron_parameter.V_syntcx: Voltage(10)
            })
        return parameters

    def measure(self, neuron_ids):
        import pycake.simulator
        params = self.get_parameters()
        results = {}
        base_freq = pycake.simulator.Simulator().compute_freq(30e-6, 0, params)

        for neuron_id in neuron_ids:
            self.sthal.switch_analog_output(neuron_id)
            self.sthal.adc.record()
            ts = np.array(self.sthal.adc.getTimestamps())
            v = np.array(self.sthal.adc.trace())

            calc_freq = lambda x: (1.0 / x - 1.0 / base_freq) * 1e6  # millions?

# TODO
#            m = sorted_array_mean[sorted_array_mean != 0.0]
#            e = sorted_array_err[sorted_array_err != 0.0]
#            return calc_freq(m), calc_freq(e)

            results[neuron_id] = calc_freq(v)
            del ts

        self.all_results.append(results)

