#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calibration of LIF parameters."""

import numpy as np
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

# Import everything needed for saving:
import pickle
import time
import os

# shorter names
Coordinate = pyhalbe.Coordinate
Enum = Coordinate.Enum
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter


class Calibrate_E_l(BaseCalibration):
    """E_l calibration."""
    target_parameter = neuron_parameter.E_l

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>50:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Is neuron spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        return np.mean(v)*1000 # Get the mean value * 1000 for mV


class Calibrate_V_t(BaseCalibration):
    """V_t calibration."""
    target_parameter = neuron_parameter.V_t

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000<5:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Neuron not spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        # Return max value. This should be more accurate than a mean value of all maxima because the ADC does not always hit the real maximum value, underestimating V_t.
        return np.max(v)*1000


class Calibrate_V_reset(BaseCalibration):
    """V_reset calibration."""
    target_parameter = shared_parameter.V_reset

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000<5:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Neuron not spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        # Return min value. This should be more accurate than a mean value of all minima because the ADC does not always hit the real minimum value, overestimating V_reset.
        return np.min(v)*1000  


class Calibrate_V_reset_shift(Calibrate_V_reset):
    """V_reset_shift calibration."""
    target_parameter = shared_parameter.V_reset

    def init_experiment(self):
        super(Calibrate_V_reset_shift, self).init_experiment()
        self.description = self.description + "Calibrate_V_reset_shift."

    def get_shared_steps(self):
        steps = []
        for voltage in self.experiment_parameters["V_reset_range"]:
            steps.append({shared_parameter.V_reset: Voltage(voltage, apply_calibration = True)})
        return defaultdict(lambda: steps)

    def process_results(self, neuron_ids):
        """ This is changed from the original processing function in order to calculate the V_reset shift."""
        # containers for final results
        results_mean = defaultdict(list)
        results_std = defaultdict(list)
        results_polynomial = {}
        results_broken = []  # will contain neuron_ids which are broken

        # self.all_results contains all measurements, need to untangle
        # repetitions and steps
        # This is done for shared and neuron parameters
        repetition = 1
        step_results = defaultdict(list)
        for result in self.all_results:
            # still in the same step, collect repetitions for averaging
            for neuron_id in neuron_ids:
                step_results[neuron_id].append(result[neuron_id])
            repetition += 1
            if repetition > self.repetitions:
                # step is done; average, store and reset
                for neuron_id in neuron_ids:
                    results_mean[neuron_id].append(HWtoDAC(np.mean(step_results[neuron_id]), shared_parameter.V_reset))
                    results_std[neuron_id].append(HWtoDAC(np.std(step_results[neuron_id]), shared_parameter.V_reset))
                repetition = 1
                step_results = defaultdict(list)
        
        # Find the shift of each neuron compared to applied V_reset:
        all_steps = self.get_shared_steps()
        for neuron_id in neuron_ids:
            results_mean[neuron_id] = np.array(results_mean[neuron_id])
            results_std[neuron_id] = np.array(results_std[neuron_id])
            steps = [HCtoDAC(step[shared_parameter.V_reset].value, shared_parameter.V_reset) for step in all_steps[neuron_id]]
            # linear fit
            m,b = np.polyfit(steps, results_mean[neuron_id], 1)
            coeffs = [b, m-1]
            # TODO find criteria for broken neurons. Until now, no broken neurons exist
            if self.isbroken(coeffs):
                results_broken.append(neuron_id)
                self.logger.INFO("Neuron {0} marked as broken with coefficients of {1:.2f} and {2:.2f}".format(neuron_id, m, b))
            self.logger.INFO("Neuron {} calibrated successfully with coefficients {}".format(neuron_id, coeffs))
            results_polynomial[neuron_id] = create_pycalibtic_polynomial(coeffs)

        self.results_mean = results_mean
        self.results_std = results_std

        # make final results available
        self.results_polynomial = results_polynomial
        self.results_broken = results_broken

    def store_results(self):
        """This base class function can be used by child classes as store_results."""
        results = self.results_polynomial
        md = pycalibtic.MetaData()
        md.setAuthor("pycake")
        md.setComment("calibration")

        logger = self.logger

        collection = self._calib_nc

        nrns = self._red_nrns

        for index in results:
            coord = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(index))
            broken = index in self.results_broken
            if broken:
                if nrns.has(coord):  # not disabled
                    nrns.disable(coord)
                    # TODO reset/delete calibtic function for this neuron
            else:  # store in calibtic
                if not collection.exists(index):
                    logger.INFO("No existing calibration data for neuron {} found, creating default dataset".format(index))
                    cal = pycalibtic.NeuronCalibration()
                    collection.insert(index, cal)
                collection.at(index).reset(21, results[index])


        self.logger.INFO("Storing calibration results")
        self.store_calibration(md)


"""
        Measurement classes start here. These classes are used to test if the calibration was successful.
        They do this by measuring with every calibration turned on.
"""

class Test_E_l(BaseTest):
    """E_l calibration."""
    target_parameter = neuron_parameter.E_l

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>50:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Is neuron spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        return np.mean(v)*1000 # Get the mean value * 1000 for mV


class Test_V_t(BaseTest):
    """V_t calibration."""
    target_parameter = neuron_parameter.V_t

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000<5:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Neuron not spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        # Return max value. This should be more accurate than a mean value of all maxima because the ADC does not always hit the real maximum value, underestimating V_t.
        return np.max(v)*1000


class Test_V_reset(BaseCalibration):
    """V_reset calibration."""
    target_parameter = shared_parameter.V_reset

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000<5:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Neuron not spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        # Return min value. This should be more accurate than a mean value of all minima because the ADC does not always hit the real minimum value, overestimating V_reset.
        return np.min(v)*1000  


class Test_g_l(BaseTest):
    target_parameter = neuron_parameter.I_gl
    pass




"""
        EVERYTHING AFTER THIS IS POINT STILL UNDER CONSTRUCTION
"""

class Calibrate_g_l(BaseCalibration):
    target_parameter = neuron_parameter.I_gl

    def init_experiment(self):
        super(Calibrate_g_l, self).init_experiment()
        self.sthal.recording_time = 1e-3
        self.stim_length = 65
        self.pulse_length = 15

        # Get the trace averager
        self.logger.INFO("{}: Creating trace averager".format(time.asctime()))
        coord_hglobal = self.sthal.hicann.index()  # grab HICANNGlobal from StHAL
        coord_wafer = coord_hglobal.wafer()
        coord_hicann = coord_hglobal.on_wafer()
        self.trace_averager = createTraceAverager(coord_wafer, coord_hicann)

    def prepare_measurement(self, step_parameters, step_id, rep_id):
        """Prepare measurement.
        Perform reset, write general hardware settings.
        """
        neuron_parameters = step_parameters[0]
        shared_parameters = step_parameters[1]

        fgc = pyhalbe.HICANN.FGControl()
        # Set neuron parameters for each neuron
        for neuron_id in self.get_neurons():
            coord = Coordinate.NeuronOnHICANN(Enum(neuron_id))
            for parameter in neuron_parameters[neuron_id]:
                value = neuron_parameters[neuron_id][parameter]
                fgc.setNeuron(coord, parameter, value)

        # Set block parameters for each block
        for block_id in range(4):
            coord = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(block_id))
            for parameter in shared_parameters[block_id]:
                value = shared_parameters[block_id][parameter]
                fgc.setShared(coord, parameter, value)

        self.sthal.hicann.floating_gates = fgc

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
            if not os.path.isdir(os.path.join(self.folder,"floating_gates")):
                os.mkdir(os.path.join(self.folder,"floating_gates"))
            pickle.dump(fgc, open(os.path.join(self.folder,"floating_gates", "step{}rep{}.p".format(step_id,rep_id)), 'wb'))

        self.sthal.write_config()


    def measure(self, neuron_ids, step_id, rep_id):
        """Perform measurements for a single step on one or multiple neurons."""
        results = {}
        for neuron_id in neuron_ids:
            self.sthal.switch_current_stimulus(neuron_id)
            t, v = self.sthal.read_adc()
            # Save traces in files:
            if self.save_traces:
                folder = os.path.join(self.folder,"traces")
                if not os.path.isdir(os.path.join(self.folder,"traces")):
                    os.mkdir(os.path.join(self.folder,"traces"))
                if not os.path.isdir(os.path.join(self.folder,"traces", "step{}rep{}".format(step_id, rep_id))):
                    os.mkdir(os.path.join(self.folder, "traces", "step{}rep{}".format(step_id, rep_id)))
                pickle.dump([t, v], open(os.path.join(self.folder,"traces", "step{}rep{}".format(step_id, rep_id), "neuron_{}.p".format(neuron_id)), 'wb'))
            results[neuron_id] = self.process_trace(t, v, neuron_id, step_id, rep_id)
        # Now store measurements in a file:
        if self.save_results:
            if not os.path.isdir(os.path.join(self.folder,"results/")):
                os.mkdir(os.path.join(self.folder,"results/"))
            pickle.dump(results, open(os.path.join(self.folder,"results/","step{}_rep{}.p".format(step_id, rep_id)), 'wb'))
        self.all_results.append(results)


    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        # Get the time between each current pulse
        dt = 129 * 4 * (self.pulse_length + 1) / self.sthal.hicann.pll_freq
        mean_trace = self.trace_averager.get_average(v, dt)[0]
        tau_m = self.trace_averager.fit_exponential(mean_trace)
        return tau_m


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

