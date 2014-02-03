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
    def get_parameters(self):
        parameters = super(Calibrate_E_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            for param, value in self.E_l_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    parameters[neuron_id][param] = value
                elif isinstance(param, shared_parameter):
                    pass
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_E_l, self).get_shared_parameters()
        for block_id in range(4):
            for param, value in self.E_l_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    pass
                elif isinstance(param, shared_parameter):
                    parameters[block_id][param] = value
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_steps(self):
        steps = []
        for voltage in self.experiment_parameters["E_l_range"]:  # 8 steps
            steps.append({neuron_parameter.E_l: Voltage(voltage),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_E_l, self).init_experiment()
        self.description = self.experiment_parameters["E_l_description"]
        self.E_l_parameters = self.experiment_parameters["E_l_parameters"]

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>50:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Is neuron spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        return np.mean(v)*1000 # Get the mean value * 1000 for mV

    def process_results(self, neuron_ids):
        super(Calibrate_E_l, self).process_calibration_results(neuron_ids, neuron_parameter.E_l, linear_fit=True)

    def store_results(self):
        super(Calibrate_E_l, self).store_calibration_results(neuron_parameter.E_l)


class Calibrate_V_t(BaseCalibration):
    """V_t calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_V_t, self).get_parameters()
        for neuron_id in self.get_neurons():
            for param, value in self.V_t_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    parameters[neuron_id][param] = value
                    if param is neuron_parameter.E_l:
                        parameters[neuron_id][param].apply_calibration = True
                elif isinstance(param, shared_parameter):
                    pass
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_V_t, self).get_shared_parameters()
        for block_id in range(4):
            for param, value in self.V_t_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    pass
                elif isinstance(param, shared_parameter):
                    parameters[block_id][param] = value
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_steps(self):
        steps = []
        for voltage in self.experiment_parameters["V_t_range"]:
            steps.append({neuron_parameter.V_t: Voltage(voltage)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_V_t, self).init_experiment()
        self.description = "Calibrate_V_t with 1000 I_pl. Calibrated E_l."
        self.V_t_parameters = self.experiment_parameters["V_t_parameters"]

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000<5:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Neuron not spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        # Return max value. This should be more accurate than a mean value of all maxima because the ADC does not always hit the real maximum value, underestimating V_t.
        return np.max(v)*1000

    def process_results(self, neuron_ids):
        super(Calibrate_V_t, self).process_calibration_results(neuron_ids, neuron_parameter.V_t, linear_fit = True)

    def store_results(self):
        super(Calibrate_V_t, self).store_calibration_results(neuron_parameter.V_t)


class Calibrate_V_reset(BaseCalibration):
    """V_reset calibration."""
    def get_parameters(self):
        parameters = super(Calibrate_V_reset, self).get_parameters()
        for neuron_id in self.get_neurons():
            for param, value in self.V_reset_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    parameters[neuron_id][param] = value
                    if param in [neuron_parameter.E_l, neuron_parameter.V_t]:
                        parameters[neuron_id][param].apply_calibration = True
                elif isinstance(param, shared_parameter):
                    pass
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_V_reset, self).get_shared_parameters()

        for block_id in range(4):
            for param, value in self.V_reset_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    pass
                elif isinstance(param, shared_parameter):
                    parameters[block_id][param] = value
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 

        return parameters

    def get_shared_steps(self):
        steps = []
        for voltage in self.experiment_parameters["V_reset_range"]:
            steps.append({shared_parameter.V_reset: Voltage(voltage)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_V_reset, self).init_experiment()
        self.description = "Calibrate_V_reset, I_pl HIGH. E_l and V_t calibrated."
        self.V_reset_parameters = self.experiment_parameters['V_reset_parameters']

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000<5:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Neuron not spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        # Return min value. This should be more accurate than a mean value of all minima because the ADC does not always hit the real minimum value, overestimating V_reset.
        return np.min(v)*1000  

    def process_results(self, neuron_ids):
        self.process_calibration_results(neuron_ids, shared_parameter.V_reset, linear_fit = True)

    def store_results(self):
        super(Calibrate_V_reset, self).store_calibration_results(shared_parameter.V_reset)


class Calibrate_V_reset_shift(Calibrate_V_reset):
    """V_reset_shift calibration."""
    def init_experiment(self):
        super(Calibrate_V_reset_shift, self).init_experiment()
        self.description = self.description + "Calibrate_V_reset_shift."
        self.V_reset_parameters = self.experiment_parameters["V_reset_parameters"]

    def get_parameters(self):
        parameters = super(Calibrate_V_reset_shift, self).get_parameters()
        for neuron_id in self.get_neurons():
            for param, value in self.V_reset_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    parameters[neuron_id][param] = value
                    if param in [neuron_parameter.E_l, neuron_parameter.V_t]:
                        parameters[neuron_id][param].apply_calibration = True
                elif isinstance(param, shared_parameter):
                    pass
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_V_reset_shift, self).get_shared_parameters()

        for block_id in range(4):
            for param, value in self.V_reset_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    pass
                elif isinstance(param, shared_parameter):
                    parameters[block_id][param] = value
                    parameters[block_id][param].apply_calibration = True
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_shared_steps(self):
        steps = []
        for voltage in self.experiment_parameters["V_reset_range"]:
            steps.append({shared_parameter.V_reset: Voltage(voltage, apply_calibration = True)})
        return defaultdict(lambda: steps)

    def process_results(self, neuron_ids):
        """This base class function can be used by child classes as process_results."""
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
    def get_parameters(self):
        parameters = super(Test_E_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            for param, value in self.E_l_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    parameters[neuron_id][param] = value
                    parameters[neuron_id][param].apply_calibration = True
                elif isinstance(param, shared_parameter):
                    pass
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_E_l, self).get_shared_parameters()
        for block_id in range(4):
            for param, value in self.E_l_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    pass
                elif isinstance(param, shared_parameter):
                    parameters[block_id][param] = value
                    parameters[block_id][param].apply_calibration = True
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_steps(self):
        steps = []
        for voltage in self.experiment_parameters["E_l_range"]:  # 8 steps
            steps.append({neuron_parameter.E_l: Voltage(voltage, apply_calibration = True),
                })
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Test_E_l, self).init_experiment()
        self.description = "TEST OF " + self.experiment_parameters["E_l_description"]
        self.E_l_parameters = self.experiment_parameters["E_l_parameters"]

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000>50:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Is neuron spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        return np.mean(v)*1000 # Get the mean value * 1000 for mV

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass


class Test_V_t(BaseTest):
    """V_t calibration."""
    def get_parameters(self):
        parameters = super(Test_V_t, self).get_parameters()
        for neuron_id in self.get_neurons():
            for param, value in self.V_t_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    parameters[neuron_id][param] = value
                    parameters[neuron_id][param].apply_calibration = True
                elif isinstance(param, shared_parameter):
                    pass
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_V_t, self).get_shared_parameters()

        for block_id in range(4):
            for param, value in self.V_t_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    pass
                elif isinstance(param, shared_parameter):
                    parameters[block_id][param] = value
                    parameters[block_id][param].apply_calibration = True
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 

        return parameters

    def get_steps(self):
        steps = []
        for voltage in self.experiment_parameters["V_t_range"]:
            steps.append({neuron_parameter.V_t: Voltage(voltage, apply_calibration = True)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Test_V_t, self).init_experiment()
        self.description = "TEST OF " + self.experiment_parameters["V_t_description"]
        self.V_t_parameters = self.experiment_parameters["V_t_parameters"]

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000<5:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Neuron not spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        # Return max value. This should be more accurate than a mean value of all maxima because the ADC does not always hit the real maximum value, underestimating V_t.
        return np.max(v)*1000

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass


class Test_V_reset(BaseCalibration):
    """V_reset calibration."""
    def get_parameters(self):
        parameters = super(Test_V_reset, self).get_parameters()
        for neuron_id in self.get_neurons():
            for param, value in self.V_reset_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    parameters[neuron_id][param] = value
                    if param in [neuron_parameter.E_l, neuron_parameter.V_t]:
                        parameters[neuron_id][param].apply_calibration = True
                elif isinstance(param, shared_parameter):
                    pass
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_shared_parameters(self):
        parameters = super(Test_V_reset, self).get_shared_parameters()

        for block_id in range(4):
            for param, value in self.V_reset_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    pass
                elif isinstance(param, shared_parameter):
                    parameters[block_id][param] = value
                    parameters[block_id][param].apply_calibration = True
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_shared_steps(self):
        steps = []
        for voltage in self.experiment_parameters["V_reset_range"]:
            steps.append({shared_parameter.V_reset: Voltage(voltage, apply_calibration = True)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Test_V_reset, self).init_experiment()
        self.description = "TEST OF " + self.experiment_parameters["V_reset_description"]
        self.V_reset_parameters = self.experiment_parameters["V_reset_parameters"]

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        if np.std(v)*1000<5:
            if not os.path.isdir(os.path.join(self.folder, "bad_traces")):
                os.mkdir(os.path.join(self.folder, "bad_traces"))
            pickle.dump([t,v], open(os.path.join(self.folder,"bad_traces","bad_trace_s{}_r{}_n{}.p".format(step_id, rep_id, neuron_id)), 'wb'))
            self.logger.WARN("Trace for neuron {} bad. Neuron not spiking? Saved to bad_trace_s{}_r{}_n{}.p".format(neuron_id, step_id, rep_id, neuron_id))
        # Return min value. This should be more accurate than a mean value of all minima because the ADC does not always hit the real minimum value, overestimating V_reset.
        return np.min(v)*1000  

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass






"""
        EVERYTHING AFTER THIS IS POINT STILL UNDER CONSTRUCTION
"""

class Calibrate_g_l(BaseCalibration):
    def init_experiment(self):
        super(Calibrate_g_l, self).init_experiment()
        self.sthal.recording_time = 1e-3
        self.description = self.experiment_parameters['g_l_description'] # Change this for all child classes
        self.g_l_parameters = self.experiment_parameters['g_l_parameters']
        self.stim_length = 65
        self.pulse_length = 15

        # Get the trace averager
        self.logger.INFO("{}: Creating trace averager".format(time.asctime()))
        coord_hglobal = self.sthal.hicann.index()  # grab HICANNGlobal from StHAL
        coord_wafer = coord_hglobal.wafer()
        coord_hicann = coord_hglobal.on_wafer()
        self.trace_averager = createTraceAverager(coord_wafer, coord_hicann)

    def get_parameters(self):
        parameters = super(Calibrate_g_l, self).get_parameters()
        for neuron_id in self.get_neurons():
            for param, value in self.g_l_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    parameters[neuron_id][param] = value
                elif isinstance(param, shared_parameter):
                    pass
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_g_l, self).get_shared_parameters()
        for block_id in range(4):
            for param, value in self.g_l_parameters.iteritems():
                if isinstance(param, neuron_parameter):
                    pass
                elif isinstance(param, shared_parameter):
                    parameters[block_id][param] = value
                else:
                    raise TypeError('Only neuron_parameter or shared_parameter allowed') 
        return parameters

    def get_steps(self):
        steps = []
        for current in self.experiment_parameters['I_gl_range']:
            steps.append({neuron_parameter.I_gl: Current(current)})
        return defaultdict(lambda: steps)

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

    def process_results(self, neuron_ids):
        self.process_calibration_results(neuron_ids, neuron_parameter.I_gl)

    def store_results(self):
        # TODO detect and store broken neurons
        super(Calibrate_g_l, self).store_calibration_results(neuron_parameter.I_gl)



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

class Calibrate_g_L_stepcurrent(BaseCalibration):
    def get_parameters(self):
        parameters = super(Calibrate_g_L_stepcurrent, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                pyhalbe.HICANN.neuron_parameter.E_l: Voltage(600, apply_calibration = True),
                pyhalbe.HICANN.neuron_parameter.V_t: Voltage(1300, apply_calibration = True),
            })
        return parameters

    def get_shared_parameters(self):
        parameters = super(Calibrate_g_L_stepcurrent, self).get_shared_parameters()
        for block_id in range(4):
            parameters[block_id][pyhalbe.HICANN.shared_parameter.V_reset] = Voltage(700, apply_calibration = True)
        return parameters

    def get_steps(self):
        steps = []
        #for current in range(700, 1050, 50):
        for current in range(700, 1300, 100):
            steps.append({pyhalbe.HICANN.neuron_parameter.I_gl: Current(current)})
        return defaultdict(lambda: steps)

    def init_experiment(self):
        super(Calibrate_g_L_stepcurrent, self).init_experiment()
        self.description = "Capacitance g_L experiment. Membrane is fitted after step current." # Change this for all child classes
        self.repetitions = 1
        self.E_syni_dist = -100
        self.E_synx_dist = +100
        self.save_results = True
        self.save_traces = False

    def prepare_measurement(self, step_parameters, step_id, rep_id):
        """Prepare measurement.
        Perform reset, write general hardware settings.
        """
        neuron_parameters = step_parameters[0]
        shared_parameters = step_parameters[1]

        fgc = pyhalbe.HICANN.FGControl()
        # Set neuron parameters for each neuron
        for neuron_id in neuron_parameters:
            coord = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(neuron_id))
            for parameter in neuron_parameters[neuron_id]:
                value = neuron_parameters[neuron_id][parameter]
                fgc.setNeuron(coord, parameter, value)
            fgstim = pyhalbe.HICANN.FGStimulus(10,10,False)
            self.sthal.hicann.setCurrentStimulus(coord, fgstim)

        # Set block parameters for each block
        for block_id in shared_parameters:
            coord = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(block_id))
            for parameter in shared_parameters[block_id]:
                value = shared_parameters[block_id][parameter]
                fgc.setShared(coord, parameter, value)

        self.sthal.hicann.floating_gates = fgc

        if self.bigcap is True:
            # TODO add bigcap functionality
            pass

        if self.save_results:
            if not os.path.isdir(os.path.join(self.folder,"floating_gates/")):
                os.mkdir(os.path.join(self.folder,"floating_gates/"))
            pickle.dump(fgc, open(os.path.join("floating_gates/","step{}rep{}.p".format(step_id,rep_id)), 'wb'))

        self.sthal.write_config()

    def process_trace(self, t, v, neuron_id, step_id, rep_id):
        return 0

    def process_results(self, neuron_ids):
        pass

    def store_results(self):
        pass


