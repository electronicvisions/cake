import pyhalbe
import numpy as np

from pycake.helpers.units import Voltage
from pycake.experiment import BaseExperiment

neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter

class E_l_I_gl_fixed_Calibrator(object):
    target_parameters = (neuron_parameter.E_l, neuron_parameter.I_gl)

    def __init__(self, experiments):
        if not isinstance(experiments, list):
            experiments = [experiments]
        self.experiment = BaseExperiment(sum(
            [exp.measurements for exp in experiments], []),
            experiments[0].analyzer, None)
        self.experiment.results = sum(
            [exp.results for exp in experiments], [])
        self.neurons = self.experiment.measurements[0].neurons
        # TODO from config? perhaps later
        self.target = 700.0
        self.min_distance = 50.0

    def fit_neuron(self, neuron):
        data = self.experiment.get_parameters_and_results(neuron,
                (neuron_parameter.E_l, neuron_parameter.I_gl), ("mean",))
        index_E_l = 0
        index_I_gl = 1
        index_mean = 2
        I_gl_steps = np.unique(data[:,index_I_gl])

        for step in I_gl_steps:
            selected = data[data[:,index_I_gl] == step]
            E_l, mean = selected[:,(index_E_l,index_mean)].T
            mean *= 1000.0

            if ((np.min(mean) < (self.target - self.min_distance))
                    and (np.max(mean) > (self.target + self.min_distance))):
                a1, a0 = np.polyfit(E_l, mean, 1)
                E_l_result = (self.target - a0)/a1 #.toDAC().value
                if E_l_result > Voltage(self.target + 300.0).toDAC().value:
                    continue
                return E_l_result, step
            else:
                continue
        return None

    def generate_coeffs(self):
        """ Takes averaged experiments and does the fits
        """
        E_l_fits = {}
        I_gl_fits = {}
        for neuron in self.neurons:
            fit = self.fit_neuron(neuron)
            if fit:
                E_l, I_gl = fit
                E_l_fits[neuron] = [E_l]
                I_gl_fits[neuron] = [I_gl]
        x = [(neuron_parameter.E_l, E_l_fits),
                (neuron_parameter.I_gl, I_gl_fits)]
        import pprint
        pprint.pprint(x[0])
        pprint.pprint(x[1])
        print len(x[0][1])
        return x
