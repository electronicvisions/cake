import pyhalbe
import numpy as np

from pycake.helpers.units import Voltage
from pycake.experiment import SequentialExperiment

neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

class E_l_I_gl_fixed_Calibrator(object):
    target_parameters = (neuron_parameter.E_l, neuron_parameter.I_gl)

    def __init__(self, experiment, config):
        self.experiment = experiment
        self.neurons = self.experiment.measurements[0].neurons
        # TODO from config? perhaps later
        self.target = config.get_config_with_default('E_l_target', 700.0)
        self.min_distance = 50.0 # mV
        self.fallback_I_gl_DAC = 1023
        # maximum deviation of DAC for target membrane voltage from ideal transformation
        self.max_delta_ideal = 300

    def fit_neuron(self, neuron):
        data = self.experiment.get_parameters_and_results(neuron,
                (neuron_parameter.E_l, neuron_parameter.I_gl), ("mean",))
        index_E_l = 0
        index_I_gl = 1
        index_mean = 2
        I_gl_DAC_steps = np.unique(data[:,index_I_gl])

        for I_gl_DAC in I_gl_DAC_steps:
            selected = data[data[:,index_I_gl] == I_gl_DAC]
            E_l, mean = selected[:,(index_E_l,index_mean)].T

            # membrane voltages (in mV) for a low and high (uncalibrated) E_l
            mean *= 1000.0

            if ((np.min(mean) < (self.target - self.min_distance))
                    and (np.max(mean) > (self.target + self.min_distance))):

                a1, a0 = np.polyfit(E_l, mean, 1)

                # DAC for target membrane voltage
                E_l_DAC_target = (self.target - a0)/a1

                # sanity check
                if E_l_DAC_target > Voltage(self.target + self.max_delta_ideal).toDAC().value:
                    continue
                return E_l_DAC_target, I_gl_DAC
            else:
                continue
        return None

    def generate_coeffs(self):
        """ Takes averaged experiments and does the fits
        """
        I_gl_fits = {}
        for neuron in self.neurons:
            fit = self.fit_neuron(neuron)
            if fit:
                E_l, I_gl_DAC = fit
                I_gl_fits[neuron] = [I_gl_DAC]
            else:
                I_gl_fits[neuron] = [self.fallback_I_gl_DAC]
        x = [(neuron_parameter.I_gl, I_gl_fits)]
        return x
