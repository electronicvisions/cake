"""Manages experiments

"""
import numpy
import numpy as np
import pylogging
from collections import defaultdict
import pyhalbe
import Coordinate
from pycake.helpers.trafos import HWtoDAC
from pycake.helpers.units import DAC


# shorter names
Enum = Coordinate.Enum
neuron_parameter = pyhalbe.HICANN.neuron_parameter
shared_parameter = pyhalbe.HICANN.shared_parameter


# TODO find better name
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

    def generate_coeffs(self):
        """ Takes averaged experiments and does the fits
        """
        coeffs = {}
        for neuron in self.get_neurons():
            # Need to switch y and x in order to get the right fit
            # (y-axis: configured parameter, x-axis: measurement)
            ys_raw, xs_raw = self.get_merged_results(neuron)
            xs = self.prepare_x(xs_raw)
            ys = self.prepare_y(ys_raw)
            coeffs[neuron] = self.do_fit(xs, ys)
            if self.is_defect(coeffs[neuron]):
                coeffs[neuron] = None
        return [(self.target_parameter, coeffs)]

    def dac_to_si(self, dac, parameter):
        """ Transforms dac value to mV or nA, depending on input parameter
        """
        if self.target_parameter.name[0] == 'I':
            return dac * 2500/1023.
        else:
            return dac * 1800/1023.

    def prepare_x(self, x):
        """ Prepares x values for fit
            Usually, this is a measured membrane voltage in V
            Per default, this function Translates from V or A to DAC
        """
        xs = [HWtoDAC(val*1000, self.target_parameter) for val in x]
        xs = np.array(xs)
        return xs

    def prepare_y(self, y):
        """ Prepares y values for fit
            Per default, these are the step (DAC) values that were set.
        """
        ys = np.array(y)
        return ys

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
        print raw_x, raw_y
        xs = self.prepare_x(raw_x)
        ys = self.prepare_y(raw_y)
        return np.polyfit(xs, ys, 1)

    def generate_coeffs(self):
        """
        Default fiting method.

        Returns:
            List of tuples, each containing the neuron parameter and a
            dictionary containing polynomial fit coefficients for each neuron
        """
        neuron_results = dict(x for x in self.iter_neuron_results())
        block_means = self.mean_over_blocks(neuron_results)

        coeffs = {}
        for coord, mean in block_means.iteritems():
            coeffs[coord] = self.do_fit(mean[:, 2], mean[:, 0])

        return [(self.target_parameter, coeffs)]


class E_synx_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.E_synx

    def is_defect(self, coeffs):
        # Defect if slope of the fit is too high or offset is significantly positive
        defect = (abs(coeffs[0]) - 1) > 1 or abs(coeffs[1]) > 100
        return defect


class E_syni_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.E_syni

    def is_defect(self, coeffs):
        # Defect if slope of the fit is too high
        defect = (abs(coeffs[0]) - 1) > 1 or abs(coeffs[1]) > 100
        return defect


class E_l_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.E_l

    """
    def is_defect(self, coeffs):
        defect = (coeffs[0] - 1) > 1 # Defect if slope of the fit is too high
        return defect
    """


class Spikes_Calibrator(BaseCalibrator):
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


class I_gl_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.I_gl

    def prepare_x(self, x):
        xs = [HWtoDAC(val, self.target_parameter) for val in x]
        xs = np.array(xs)
        return xs

    def get_key(self):
        return 'tau_m'


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

    def generate_coeffs(self):

        coeffs = {}

        for neuron in self.neurons:
            V_syntc_psp_max = self.fit_neuron(neuron)
            coeffs[neuron] = [V_syntc_psp_max]

        return [(self.target_parameter, coeffs)]

    """
    def is_defect(self, coeffs, xs):
        defect = np.max(xs) < 0.005
        print xs, np.max(xs), defect
    """


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

    def generate_coeffs(self):
        """
        Default fiting method.

        Returns:
            List of tuples, each containing the neuron parameter and a
            dictionary containing polynomial fit coefficients for each neuron
        """
        coeffs = dict((n, self.find_v_convoff(n)) for n in self.get_neurons())
        return [(self.target_parameter, coeffs)]

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
            return self.find_roots(x, f)
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

    def prepare_x(self, x):
        """ Prepares x values for fit
            Here, the refractory period needs to be calculated from dt of spikes.
            The last measurement is the one with I_pl=2500, so tau_ref=0
        """
        dts = np.array(x)
        tau_refracs = dts - dts[-1]
        tau_refracs = tau_refracs[0:-1]
        xs = HWtoDAC(tau_refracs, self.target_parameter)
        return xs

    def prepare_y(self, y):
        """ Prepares y values for fit
            Per default, these are the step (DAC) values that were set.
        """
        ys = np.array(y)
        return ys[0:-1]


class readout_shift_Calibrator(BaseCalibrator):
    def __init__(self, experiments, config, neuron_size=64):
        super(readout_shift_Calibrator, self).__init__(experiments, config)
        self.neuron_size = neuron_size

    def generate_coeffs(self):
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
        n_blocks = 512/self.neuron_size # no. of interconnected neurons
        readout_shifts = {}
        for block_id in range(n_blocks):
            neurons_in_block = self.get_neuron_block(block_id, self.neuron_size)
            V_rests_in_block = np.array([results[neuron]['mean'] for neuron in neurons_in_block])
            mean_V_rest_over_block = np.mean(V_rests_in_block)
            for neuron in neurons_in_block:
                readout_shifts[neuron] = [float(results[neuron]['mean'] - mean_V_rest_over_block)]
        return readout_shifts


class I_gladapt_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.I_gladapt
