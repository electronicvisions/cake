import numpy as np
import os

import Coordinate

from pycake.measure import ADCMeasurement
from pycake.analyzer import PeakAnalyzer
from pycake.experimentbuilder import BaseExperimentBuilder
from pycake.experiment import SequentialExperiment
from pycake.calibrator import BaseCalibrator
from pycake.helpers.WorkerPool import WorkerPool
from pycake.helpers.misc import mkdir_p


class Capacitance_Measurement(ADCMeasurement):
    def __init__(self, sthal, neurons, readout_shifts=None, current_dacs=np.arange(50, 1000, 100)):
        super(Capacitance_Measurement, self).__init__(sthal, neurons, readout_shifts)
        self.current_dacs = current_dacs

    def pre_measure(self, neuron, current_dac, stim_length=129):
        """ Set current injection to value current_dac
            Firing is activated
        """
        if current_dac is None:
            # No firing activated
            self.sthal.switch_analog_output(neuron)
        else:
            self.sthal.set_current_stimulus(int(current_dac), stim_length)
            self.sthal.switch_current_stimulus_and_output(neuron)

    def _measure(self, analyzer, additional_data):
        """ Measure traces and correct each value for readout shift.
            Also applies analyzer to measurement
        """
        self.logger.INFO("Measuring.")
        worker = WorkerPool(analyzer)
        for neuron in self.neurons:
            self.logger.TRACE("Measuring neuron {}".format(neuron))
            if not self.traces is None:
                self.traces[neuron] = {}
            for current_dac in self.current_dacs:
                self.pre_measure(neuron, current_dac)
                readout = self.read_adc()
                readout['current_dac'] = current_dac
                readout['pll_freq'] = self.sthal.getPLL()
                if not self.traces is None:
                    self.traces[neuron] = readout
                readout['v'] = self.readout_shifts(neuron, readout['v'])
                worker.do((neuron, current_dac), neuron=neuron, **readout)
        self.logger.INFO("Wait for analysis to complete.")
        results = worker.join()
        results_sorted = self.sort_results(results)
        return results_sorted

    def sort_results(self, results):
        """ Sorts the result dictionary.
            It is nicer to have the results in the following format:
            {neuron: [result1, result2, ...], ...}
            instead of
            {(neuron, current1): result1, (neuron, current2): result2, ...}
        """
        results_sorted = {}
        for neuron in self.neurons:
            neuron_result = [results[(neuron, current_dac)] for current_dac in self.current_dacs]
            results_sorted[neuron] = neuron_result
        return results_sorted

class Capacitance_Analyzer(PeakAnalyzer):
    def __init__(self, save_peaks=False):
        super(Capacitance_Analyzer, self).__init__()
        self.save_peaks = save_peaks

    def __call__(self, neuron, t, v, **traces):
        maxtab, mintab = self.get_peaks(t, v)
        pll = traces['pll_freq']
        hard_max, hard_min = np.max(v), np.min(v)
        mean_trace, std_trace = np.mean(v), np.std(v)
        # The current from the current generator is approximately:
        # I = Voltage * 1.95 uA / V = DAC * 1.8/1023. * 1.95
        # --> See Millner thesis
        current_I = traces['current_dac'] * 1.8/1023. * 1.95
        if len(maxtab) > 0 and len(mintab>0):
            n_spikes = len(maxtab)
            total_time = (maxtab[-1][0]*1/pll)
            l = min([len(maxtab), len(mintab)])
            # Remove the times it takes for resetting. This is significant for high frequencies!
            mean_reset_time = abs(np.mean(mintab[:,0][:l] - maxtab[:,0][:l])/pll)
            total_time_without_reset = total_time - mean_reset_time * n_spikes
            freq = n_spikes / total_time_without_reset
            mean_max = np.mean(maxtab[:, 1])
            std_max = np.std(maxtab[:, 1])
            mean_min = np.mean(mintab[:, 1])
            std_min = np.std(mintab[:, 1])
            amplitude = mean_max - mean_min
        else:
            # If get_peaks didn't work, return nans
            mean_max, mean_min, std_max, std_min, freq = np.nan, np.nan, np.nan, np.nan, np.nan
            amplitude = hard_max - hard_min

        results = {"hard_max": hard_max,
                   "hard_min": hard_min,
                   "mean": mean_trace,
                   "std": std_trace,
                   "mean_max": mean_max,
                   "mean_min": mean_min,
                   "mean_max": std_max,
                   "mean_min": std_min,
                   "frequency": freq,
                   "current": current_I,
                   "amplitude": amplitude}

        if self.save_peaks:
            results["maxtab"] = maxtab
            results["mintab"] = mintab

        return results


class Capacitance_Experimentbuilder(BaseExperimentBuilder):
    def __init__(self, config, test=False):
        self.config = config

        path, name = self.config.get_calibtic_backend()
        wafer, hicann = self.config.get_coordinates()

        self.neurons = self.config.get_neurons()
        self.blocks = self.config.get_blocks()
        self.test = test
        if test:
            raise NotImplementedError("Capacitance cannot be calibrated, so test measurement cannot be done")

        self.measurements_folder = 'measurements/'

    def get_readout_shifts(self, neurons):
        """ Get readout shifts (in V) for a list of neurons.
            If no readout shift is saved, a shift of 0 is returned.

            Args:
                neurons: a list of NeuronOnHICANN coordinates
            Returns:
                shifts: a dictionary {neuron: shift (in V)}
        """
        return dict((n, 0) for n in neurons)

    def prepare_specific_config(self, sthal):
        """ 
        """
        if self.config.parameters['smallcap']:
            sthal.hicann.use_big_capacitors(False)
        return sthal

    def make_measurement(self, sthal, neurons, readout_shifts):
        return Capacitance_Measurement(sthal, neurons, readout_shifts)

    def get_analyzer(self):
        return Capacitance_Analyzer()

    def get_experiment(self, config_name):
        """
        """
        measurements, repetitions = self.generate_measurements()
        analyzer = self.get_analyzer()
        experiment = SequentialExperiment(measurements, analyzer, repetitions)
        #if self.config.get_save_traces():
        #    experiment.save_traces(
        #        self.get_measurement_storage_path(config_name))
        return experiment

    def get_measurement_storage_path(self, config_name):
        """ Save measurement i of experiment to a file and clear the traces from
            that measurement.
        """
        folder = os.path.join(
            self.config.get_folder(),
            self.measurements_folder,
            config_name)
        mkdir_p(folder)
        return folder


class Capacitance_Calibrator(BaseCalibrator):
    def generate_coeffs(self):
        """ Calculates Capacitance from measurements
        """
        all_results = self.experiments[0].results[0]
        caps = {}
        for neuron, results in all_results.iteritems():
            fs = np.array([r['frequency'] for r in results])
            Is = np.array([r['current'] for r in results])
            As = np.array([r['amplitude'] for r in results])
            fAs = fs * As
            fit_results = np.polyfit(Is[1:-3], fAs[1:-3], 1)
            slope_of_fit = fit_results[0]
            C = 1/slope_of_fit
            caps[neuron] = C
        return [('capacitance', caps)]
