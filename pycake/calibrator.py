"""Manages experiments

"""
import numpy
import numpy as np
from scipy.optimize import curve_fit
import pylogging
from collections import defaultdict
from pyhalbe.HICANN import neuron_parameter, shared_parameter
import Coordinate
import pycalibtic
from pycake.helpers.calibtic import create_pycalibtic_transformation

class BaseCalibrator(object):
    """ Takes experiments and a target parameter and turns this into calibration data via a linear fit.
        Does NOT save any data.

    Args:
        target_parameter: neuron_parameter or shared_parameter
        feature: string containing the feature key (e.g. "mean" for E_l or E_syn, "g_l" for I_gl, ...)
                 this is used to extract the results that were given by the analyzers of an experiment
        experiments: list containing all experiments
    """
    logger = pylogging.get("pycake.calibrator")

    def __init__(self, experiments, config):
        if not isinstance(experiments, list):
            experiments = [experiments]
        self.experiments = experiments

    def get_key(self):
        """
        """
        return 'mean'

    def check_for_same_config(self, measurements):
        """ Checks if measurements have the same config.
            Only the target parameter is allowed to be different.
        """
        # TODO implement checking
        return True

    def get_step_parameters(self, measurement):
        """
        """
        step_parameters = measurement.get_parameter(
            self.target_parameter, measurement.neurons)
        return step_parameters

    def get_merged_results(self, neuron):
        """ Return all parameters and results from all experiments for one neuron.
            This does not average over experiments.
        """
        results = []
        for ex in self.experiments:
            results.append(ex.get_parameters_and_results(neuron, [self.target_parameter], [self.get_key()]))
        # return 'unzipped' results, e.g.: (100,0.1),(200,0.2) -> (100,200), (0.1,0.2)
        return zip(*np.concatenate(results))

    def generate_transformations(self):
        """ Takes averaged experiments and does the fits
        """
        transformations = {}
        trafo_type = self.get_trafo_type()

        for neuron in self.get_neurons():
            # Need to switch y and x in order to get the right fit
            # (y-axis: configured parameter, x-axis: measurement)
            ys_raw, xs_raw = self.get_merged_results(neuron)
            xs = self.prepare_x(xs_raw)
            ys = self.prepare_y(ys_raw)
            # Extract function coefficients and domain from measured data
            coeffs = self.do_fit(xs, ys)
            coeffs = coeffs[::-1] # coeffs are reversed in calibtic transformations
            domain = self.get_domain(xs)
            transformations[neuron] = create_pycalibtic_transformation(coeffs, domain, trafo_type)
            if self.is_defect(coeffs):
                transformations[neuron] = None
        return [(self.target_parameter, transformations)]

    def get_domain(self, data):
        """ Extract the domain from measured data.
            Default always returns None
        """
        return None

    def dac_to_si(self, dac, parameter):
        """ Transforms dac value to mV or nA, depending on input parameter
        """
        if self.target_parameter.name[0] == 'I':
            return dac * 2500/1023.
        else:
            return dac * 1.8/1023.

    def prepare_x(self, x):
        """ Prepares x values for fit
            This should be the raw measured hardware data
        """
        return np.array(x)

    def prepare_y(self, y):
        """ Prepares y values for fit
            Per default, these are the step (DAC) values that were set.
        """
        return np.array(y)

    def do_fit(self, xs, ys):
        """ Fits a curve to results of one neuron
            Standard behaviour is a linear fit.
        """
        fit_coeffs = np.polyfit(xs, ys, 1)
        return fit_coeffs

    def is_defect(self, coeffs):
        return False

    def get_neurons(self):
        # TODO: Improve this
        return self.experiments[0].measurements[0].get_neurons()

    def get_trafo_type(self):
        """ Returns the pycalibtic.transformation type that is used for calibration.
            Default returns Polynomial
        """
        return pycalibtic.Polynomial


class V_reset_Calibrator(BaseCalibrator):
    target_parameter = shared_parameter.V_reset

    def __init__(self, experiment, config):
        self.experiment = experiment

    def get_key(self):
        return 'baseline'

    def iter_neuron_results(self):
        key = self.get_key()
        for neuron in self.get_neurons():
            baseline, = self.experiment.get_mean_results(
                neuron, self.target_parameter, (key, ))
            yield neuron, baseline

    def mean_over_blocks(self, neuron_results):
        blocks = defaultdict(list)
        for neuron, value in neuron_results.iteritems():
            blocks[neuron.toSharedFGBlockOnHICANN()].append(value)
        return dict((k, np.mean(v, axis=0)) for k, v in blocks.iteritems())

    def get_neurons(self):
        return self.experiment.measurements[0].neurons

    def do_fit(self, raw_x, raw_y):
        xs = self.prepare_x(raw_x)
        ys = self.prepare_y(raw_y)
        return np.polyfit(xs, ys, 1)

    def generate_transformations(self):
        """
        Default fiting method.

        Returns:
            List of tuples, each containing the neuron parameter and a
            dictionary containing polynomial fit coefficients for each neuron
        """
        neuron_results = dict(x for x in self.iter_neuron_results())
        block_means = self.mean_over_blocks(neuron_results)

        trafos = {}
        trafo_type = self.get_trafo_type()
        for coord, mean in block_means.iteritems():
            fit = list(self.do_fit(mean[:, 2], mean[:, 0]))
            fit = fit[::-1] # reverse coefficients for calibtic
            domain = self.get_domain(mean[:,2])
            trafo = create_pycalibtic_transformation(fit, domain, trafo_type)
            trafos[coord] = trafo


        return [(self.target_parameter, trafos)]

    def get_domain(self, data):
        return [0,1.8]


class E_synx_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.E_synx

    def get_domain(self, data):
        return [0,1.8]

    #def is_defect(self, coeffs):
    #    # Defect if slope of the fit is too high or offset is significantly positive
    #    defect = (abs(coeffs[0]) - 1) > 1 or abs(coeffs[1]) > 100
    #    return defect


class E_syni_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.E_syni

    def get_domain(self, data):
        return [0,1.8]

    #def is_defect(self, coeffs):
    #    # Defect if slope of the fit is too high
    #    defect = (abs(coeffs[0]) - 1) > 1 or abs(coeffs[1]) > 100
    #    return defect


class E_l_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.E_l

    """
    def is_defect(self, coeffs):
        defect = (coeffs[0] - 1) > 1 # Defect if slope of the fit is too high
        return defect
    """
    def get_domain(self, data):
        return [0,1.8]


class spikes_Calibrator(BaseCalibrator):
    target_parameter = None


class InputSpike_Calibrator(BaseCalibrator):
    target_parameter = None


class V_t_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.V_t

    def get_key(self):
        return 'max'

    def is_defect(self, coeffs):
        # Defect if slope of the fit is too high
        defect = abs(coeffs[0] - 1) > 1
        return defect

    def get_domain(self, data):
        return [0,1.8]

class I_gl_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.I_gl

    def prepare_x(self, x):
        xs = np.array(xs)
        return xs

    def get_key(self):
        return 'tau_m'

    # TODO: implement proper domain detection
    def get_domain(self, data):
        return [0,2.5e-6]

    def get_trafo_type(self):
        """ Returns the pycalibtic.transformation type that is used for calibration.
            Default returns Polynomial
        """
        return pycalibtic.NegativePowersPolynomial

class I_gl_charging_Calibrator(I_gl_Calibrator):
    pass


from pycake.calibration.E_l_I_gl_fixed import E_l_I_gl_fixed_Calibrator


class V_syntc_psp_max_BaseCalibrator(BaseCalibrator):

    # to be set in derived class
    target_parameter = None

    def __init__(self, experiment, config=None):
        self.experiment = experiment
        self.neurons = self.experiment.measurements[0].neurons

    def get_key(self):
        return 'std'

    def fit_neuron(self, neuron):
        data = self.experiment.get_parameters_and_results(neuron,
                                                          [self.target_parameter],
                                                          ["std"])
        index_V_syntc = 0
        index_std = 1

        V_syntc_steps = np.unique(data[:, index_V_syntc])

        V_syntc_psp_max = 0
        max_std = 0

        for step in V_syntc_steps:
            selected = data[data[:, index_V_syntc] == step]
            V_syntc, std = selected[:, (index_V_syntc, index_std)].T

            #V_syntc = V_syntc[0]
            #std = std[0]

            if std > max_std:
                V_syntc_psp_max = V_syntc[0]
                max_std = std[0]

        return V_syntc_psp_max

    def generate_transformations(self):

        trafos = {}

        for neuron in self.neurons:
            V_syntc_psp_max = self.fit_neuron(neuron)
            # For compatibility, the domain should be None
            trafo = create_pycalibtic_transformation([V_syntc_psp_max], None)
            trafos[neuron] = trafo

        return [(self.target_parameter, trafos)]


class V_syntci_psp_max_Calibrator(V_syntc_psp_max_BaseCalibrator):
    target_parameter = neuron_parameter.V_syntci


class V_syntcx_psp_max_Calibrator(V_syntc_psp_max_BaseCalibrator):
    target_parameter = neuron_parameter.V_syntcx


class V_convoff_Calibrator(BaseCalibrator):
    target_parameter = None
    MEMBRANE_SHIFT = None

    def __init__(self, experiment, config=None):
        self.experiment = experiment

    def get_neurons(self):
        return self.experiment.measurements[0].neurons

    def generate_transformations(self):
        """
        Default fiting method.

        Returns:
            List of tuples, each containing the neuron parameter and a
            dictionary containing polynomial fit coefficients for each neuron
        """
        trafos = dict((n, self.find_v_convoff(n)) for n in self.get_neurons())
        return [(self.target_parameter, trafos)]

    def get_fit_data(self, neuron):
        parameters = (self.target_parameter, )
        data = self.experiment.get_parameters_and_results(
                neuron, parameters, ('baseline', ))
        V_convoff, baseline = data.T

        # normalize area by distance to reversal potential
        # area = area / (E_synx / 1023 * 1.8 - baseline)

        return (
            V_convoff,
            baseline - baseline[-1],
        )

    def do_fit(self, V_convoff, baseline):
        """
        Fits a polynomial of the 5th degree to catch strange tails from
        PSPs near the reversale potential...
        """
        # Fit to the linear rising part
        d = numpy.abs(baseline - baseline[-1])
        idx = d > 2.0e-3
        # Add leftmoste removed point to fit
        idx[numpy.where(idx == False)[0][0]] = True
        x = V_convoff[idx]
        y = baseline[idx]
        return x, numpy.poly1d(numpy.polyfit(x, y, 5))

    def find_roots(self, V_convoff, f):
        """Filter real solutions in input range"""
        f -= numpy.poly1d([self.MEMBRANE_SHIFT])
        r = f.roots[numpy.isreal(f.roots)]
        r = r[(V_convoff[0] < r) & (r < V_convoff[-1])]
        return r.real

    def find_v_convoff(self, neuron):
        V_convoff, baseline = self.get_fit_data(neuron)
        try:
            x, f = self.do_fit(V_convoff, baseline)
            ret = create_pycalibtic_transformation(self.find_roots(x,f), None)
            return ret
        except TypeError:
            return None

    def debug_plot(self, neuron):
        pass


class V_convoffi_Calibrator(V_convoff_Calibrator):
    target_parameter = neuron_parameter.V_convoffi
    MEMBRANE_SHIFT = -10.0e-3


class V_convoffx_Calibrator(V_convoff_Calibrator):
    target_parameter = neuron_parameter.V_convoffx
    MEMBRANE_SHIFT = 10.0e-3


class I_pl_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.I_pl

    def get_key(self):
        return 'mean_isi'

    def get_merged_results(self, neuron, key):
        """ Return all parameters and results from all experiments for one neuron.
            This does not average over experiments.
        """
        results = []
        for ex in self.experiments:
            results.append(ex.get_parameters_and_results(neuron, [self.target_parameter], key))
        # return 'unzipped' results, e.g.: (100,0.1),(200,0.2) -> (100,200), (0.1,0.2)
        return zip(*np.concatenate(results))

    def generate_transformations(self):
        """ Takes averaged experiments and does the fits
        """
        transformations = {}
        trafo_type = self.get_trafo_type()

        for neuron in self.get_neurons():
            # Need to switch y and x in order to get the right fit
            # (y-axis: configured parameter, x-axis: measurement)
            ys_raw, xs_raw, amplitudes = self.get_merged_results(neuron, ['mean_isi', 'amplitude'])
            tau0_indices = np.array(ys_raw) == 1023
            ys = self.prepare_y(ys_raw, tau0_indices)
            if len(tau0_indices) is 0:
                raise("No I_pl=2.5 uA measurement done. Cannot calculate tau_ref!")
            xs = self.prepare_x(xs_raw, amplitudes, tau0_indices)
            # Extract function coefficients and domain from measured data
            coeffs = self.do_fit(xs, ys)
            coeffs = coeffs[::-1] # coeffs are reversed in calibtic transformations
            domain = self.get_domain(xs)
            transformations[neuron] = create_pycalibtic_transformation(coeffs, domain, trafo_type)
            if self.is_defect(coeffs):
                transformations[neuron] = None
        return [(self.target_parameter, transformations)]

    def prepare_x(self, x, amplitudes, tau0_indices):
        """ Prepares x values for fit
            Here, the refractory period needs to be calculated from dt of spikes.
            tau_ref is calculated via ISI_n - ISI_0, where ISI_0 is the mean ISI of
            all measurements with I_pl = 1023 DAC. Correction for amplitude variations
            are also done, resulting the final calculation of tau_ref for step n:
                tau_ref_n = ISI_n - amplitude_n / amplitude_0 * ISI_0
        """
        x = np.array(x)
        amplitudes = np.array(amplitudes)
        tau0 = np.mean(x[tau0_indices])
        mean_amp0 = np.mean(amplitudes[tau0_indices])
        tau0amps = amplitudes * tau0/mean_amp0
        tau_refracs = x - tau0amps
        tau_ref_indices = tau0_indices == False
        return tau_refracs[tau_ref_indices]

    def prepare_y(self, y, tau0_indices):
        """ Prepares y values for fit
            Per default, these are the step (DAC) values that were set.
        """
        ys = np.array(y)
        tau_ref_indices = tau0_indices == False
        return ys[tau_ref_indices]

    def get_domain(self, data):
        # assume up to 20% higher possible domain than max measured value
        return [0, max(data) * 1.20]

    def do_fit(self, xs, ys):
        """ Fits a curve to results of one neuron
            For I_pl, the function is I_pl_DAC = 1/(a*tau + 1/1023.)
        """
        def func(x, a):
            return 1/(a*x + 1/1023.)
        fit_coeffs = curve_fit(func, xs, ys, [0.025e6])[0]
        fit_coeffs = [fit_coeffs[0], 1/1023.]
        return fit_coeffs

    def get_trafo_type(self):
        """ Returns the pycalibtic.transformation type that is used for calibration.
            Default returns Polynomial
        """
        return pycalibtic.OneOverPolynomial

class readout_shift_Calibrator(BaseCalibrator):
    def __init__(self, experiments, config, neuron_size=64):
        super(readout_shift_Calibrator, self).__init__(experiments, config)
        self.neuron_size = neuron_size

    def generate_transformations(self):
        """
        """
        results = self.experiments[0].results[0]
        readout_shifts = self.get_readout_shifts(results)
        return [('readout_shift', readout_shifts)]

    def get_neuron_block(self, block_id, size):
        if block_id * size > 512:
            raise ValueError, "There are only {} blocks of size {}".format(512/size, size)
        nids_top = np.arange(size/2*block_id,size/2*block_id+size/2)
        nids_bottom = np.arange(256+size/2*block_id,256+size/2*block_id+size/2)
        nids = np.concatenate([nids_top, nids_bottom])
        neurons = [Coordinate.NeuronOnHICANN(Coordinate.Enum(int(nid))) for nid in nids]
        return neurons

    def get_readout_shifts(self, results):
        """
        """
        trafo_type = self.get_trafo_type()
        n_blocks = 512/self.neuron_size # no. of interconnected neurons
        readout_shifts = {}
        for block_id in range(n_blocks):
            neurons_in_block = self.get_neuron_block(block_id, self.neuron_size)
            V_rests_in_block = np.array([results[neuron]['mean'] for neuron in neurons_in_block])
            mean_V_rest_over_block = np.mean(V_rests_in_block)
            for neuron in neurons_in_block:
                # For compatibility purposes, domain should be None
                shift = float(results[neuron]['mean'] - mean_V_rest_over_block)
                trafo = create_pycalibtic_transformation([shift], None, trafo_type)
                readout_shifts[neuron] = trafo
        return readout_shifts


class I_gladapt_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.I_gladapt
