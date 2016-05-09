"""Manages experiments

"""
import os
import cPickle
import numpy
import pandas
import numpy as np
from scipy.optimize import curve_fit
import pylogging
from collections import defaultdict
from pyhalbe.HICANN import neuron_parameter, shared_parameter
import Coordinate
import pycalibtic
from pycake.helpers.calibtic import create_pycalibtic_transformation
from pycalibtic import Constant
from lmfit import Parameters, minimize, fit_report, models
from pycake.helpers.units import Volt

def is_defect_potential(slope, offset, slope_from_one=1, offset_cut=1000):
    """

    returns true if slope is not close to one or offset is too large
    assuming that a trivial V = slope*DAC/1023*1.8 + offset transformation holds

    slope: DAC/Volt
    offset: DAC

    """

    slope_not_close_to_one = abs(slope/1023.*1.8 - 1) > slope_from_one
    offset_too_large = abs(offset) > offset_cut

    defect =  slope_not_close_to_one or offset_too_large

    return defect

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
    target_parameter = None

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
            if coeffs is None:
                self.logger.WARN("Fit failed for neuron {}".format(neuron))
                transformations[neuron] = None
                continue
            coeffs = coeffs[::-1] # coeffs are reversed in calibtic transformations
            domain = self.get_domain(xs)
            transformations[neuron] = create_pycalibtic_transformation(coeffs, domain, trafo_type)
            if self.is_defect(coeffs, domain):
                self.logger.WARN("Neuron {} with coeffs {} and domain {} marked as defect".format(neuron, coeffs, domain))
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

        has to return None if fit fails

        """

        ok = self.sanity_check_fit_input(xs, ys)

        if ok == False:
            return None
        else:
            fit_coeffs = np.polyfit(xs, ys, 1)
            return fit_coeffs

    def sanity_check_fit_input(self, xs, ys):
        """

        basic sanity checks of fit input

        returns True if xs and ys are "ok" and False otherwise
        """

        ok = True

        if len(xs) != len(ys):
            self.logger.DEBUG("sanity_check_fit_input: len(xs) != len(ys)")
            ok = False

        if len(xs) == 0:
            self.logger.DEBUG("sanity_check_fit_input: len(xs) == 0")
            ok = False

        if len(ys) == 0:
            self.logger.DEBUG("sanity_check_fit_input: len(ys) == 0")
            ok = False

        if ok == False:
            self.logger.WARN("Fit input did not pass sanity checks! xs: {}, ys: {}".format(xs, ys))

        return ok

    def is_defect(self, coeffs, domain):
        """ Returns True if an entry in domain or coeffs is nan.
        """
        if len(np.where(np.isnan(coeffs))[0])>0 or len(np.where(np.isnan(domain))[0])>0:
            return True
        else:
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

    def is_defect(self, coeffs, domain):
        return is_defect_potential(coeffs[1], coeffs[0])

class E_syni_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.E_syni

    def get_domain(self, data):
        return [0,1.8]

    def is_defect(self, coeffs, domain):
        return is_defect_potential(coeffs[1], coeffs[0])

class E_l_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.E_l

    def get_domain(self, data):
        return [0,1.8]

    def is_defect(self, coeffs, domain):
        return is_defect_potential(coeffs[1], coeffs[0], slope_from_one=1.5)


class Spikes_Calibrator(BaseCalibrator):
    target_parameter = None


class InputSpike_Calibrator(BaseCalibrator):
    target_parameter = None


class V_t_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.V_t

    def get_key(self):
        return 'max'

    def get_domain(self, data):
        return [0,1.8]

    def is_defect(self, coeffs, domain):
        return is_defect_potential(coeffs[1], coeffs[0])

class I_gl_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.I_gl

    def prepare_x(self, xs):
        return np.array(xs)

    def get_key(self):
        return 'tau_m'

    def get_domain(self, data):
        return [np.nanmin(data), np.nanmax(data)]

    def get_trafo_type(self):
        """ Returns the pycalibtic.transformation type that is used for calibration.
            Default returns Polynomial
        """
        return pycalibtic.NegativePowersPolynomial

    def do_fit(self, xs, ys):
        """ Fits a curve to results of one neuron
            Standard behaviour is a linear fit.
        """
        # coefficients will be reversed afterwards
        def func(x, a, b):
            return a*1/x + b*1/x**2
        try:
            fit_coeffs = curve_fit(func, xs, ys, [-82.6548e-6, 248.6586e-12])[0]
            fit_coeffs = [fit_coeffs[1], fit_coeffs[1], 0]
        except ValueError, e:
            self.logger.WARN("Could not fit results of I_gl because: {}".format(e))
            fit_coeffs = None
        return fit_coeffs

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

class V_convoff_Calibrator(BaseCalibrator):
    target_parameter = None
    v_range=0.150

    def __init__(self, experiment, config=None):
        self.experiment = experiment

    @staticmethod
    def f(params, x):
        a = params['a'].value
        b = params['b'].value
        c = params['c'].value
        return numpy.clip(a * (x - b) + c, c, numpy.inf)

    @staticmethod
    def residual(params, x, data, eps_data):
        model = V_convoff_Calibrator.f(params, x)
        return (data - model) / eps_data

    def prepare_data(self, data, v_range, spiking_threshold):
        raise NotImplemented(self)

    def find_optimum(self, data, v_range=0.150, spiking_threshold=0.001):

        fit_data = self.prepare_data(data.copy(), v_range, spiking_threshold)
        if len(fit_data[0]) < 4:
            return fit_data, None, None, None

        results = []
        params = Parameters()
        params.add('a', value=0.0)
        params.add('b', value=fit_data[0][len(fit_data)/2])
        params.add('c', value=0.0)

        out = minimize(self.residual, params,
                       args=(fit_data[0], fit_data[1], 1.0))
        return fit_data, numpy.sum(out.residual**2), out, params

    def plt_fits(self, axis, results, legend=False):
        fit_data, res, out, params = results
        x, y = fit_data
        axis.plot(x, y, color='k')
        # print fit_report(out)
        #axis.plot(
        #    x, f(params, x), 'x', 
        #    label='Res**2: {:.2f}, b: {:.2f}, chi2: {:.5f}'.format(
        #        res, params['b'].value, out.chisqr))
        if params is not None:
            x = numpy.linspace(0, 1.0, 100)
            l = axis.plot(x, self.f(params, x), '--',
                          label='b: {:.2f}'.format(params['b'].value))
            axis.plot([params['b'].value], [params['c'].value], 'x',
                      color=l[0].get_color())
        if legend:
            axis.legend(loc='lower right')

    def plt_residuals(self, axis, results):
        axis.plot([res for res, _, _ in results])

    @staticmethod
    def V_convoff(results):
        _, _, _, params = results
        if params is not None:
            return params['b'].value * 1023

    def plot_fit_for_neuron(self, nrn, axis, v_range=None):
        if v_range is None:
            v_range = self.v_range
        data = self.experiment.get_data(
            nrn, (self.target_parameter,), ('mean', 'std'))
        results = self.find_optimum(
            data, v_range, self.get_spiking_threshold(nrn))
        self.plt_fits(axis, results)

    def get_spiking_threshold(self, nrn):
        """
        Get the the threshold for considering a neuron spiking
        """
        return self.experiment.initial_data[nrn].get('std', 0.001) * 1.2

    def generate_transformations(self):
        fits = {}
        data = self.experiment.get_all_data(
            (self.target_parameter,), ('mean', 'std'))
        for nrn, data in data.groupby(level='neuron'):
            results = self.find_optimum(
                data, self.v_range, self.get_spiking_threshold(nrn))
            value = self.V_convoff(results)
            fits[nrn] = Constant(value) if value is not None else None
        return [(self.target_parameter, fits)]


class V_convoffi_Calibrator(V_convoff_Calibrator):
    target_parameter = neuron_parameter.V_convoffi
    v_range=0.120

    def prepare_data(self, data, v_range, spiking_threshold):
        """scales data from -1.0 to 0.0 and removes spiking traces"""
        data['V_convoffi'] /= 1023
        idx = (data['std'] < spiking_threshold)  # not spiking
        if not idx.any():
            self.logger.WARN("V_convoffi: Neuron considered spiking for all set V_convoffi values.")
            return numpy.empty((2, 0))

        data['mean'] = (data['mean'][idx].max() - data['mean']) / v_range
        idx &= (data['mean'] <= 1.0)

        return data[['V_convoffi', 'mean']][idx].values.T


class V_convoffx_Calibrator(V_convoff_Calibrator):
    target_parameter = neuron_parameter.V_convoffx

    def prepare_data(self, data, v_range, spiking_threshold):
        """scales data from 0.0 to 1.0 and removes spiking traces"""
        data['V_convoffx'] /= 1023
        idx = (data['std'] < spiking_threshold)  # not spiking
        if not idx.any():
            self.logger.WARN("V_convoffx: Neuron considered spiking for all set V_convoffx values.")
            return numpy.empty((2, 0))

        data['mean'] = (data['mean'] - data['mean'][idx].min())/v_range
        idx &= (data['mean'] <= 1.0)
        return data[['V_convoffx', 'mean']][idx].values.T


class V_syntc_Calibrator(BaseCalibrator):
    target_parameter = None

    def __init__(self, experiment, config=None):
        self.experiment = experiment
        self.offset_tol = 0.025
        self.chi2_tol = 30
        self.residual_tol = 4.0e-8

    @staticmethod
    def sort_tau(data):
        tmp = data[['tau_1', 'tau_2']]
        data['tau_1'] = tmp.min(axis='columns')
        data['tau_2'] = tmp.max(axis='columns')

    def get_mask(self, data):
        upper_threshold = data['offset'].min() + self.offset_tol
        return ((data['offset'] <= upper_threshold) &
                (data['chi2'] <= self.chi2_tol))

    def get_data(self):
        data = self.experiment.get_all_data(
            (self.target_parameter, ),
            ('v', 'tau_1', 'tau_2', 'start', 'offset', 'chi2'))
        data.rename(columns={self.target_parameter.name: 'V_syntc'},
                    inplace=True)
        self.sort_tau(data)
        return data

    def approximate_with_fit(self, nrn_data):
        mask = self.get_mask(nrn_data)
        data = nrn_data[mask]

        vsyntc = data['V_syntc'].astype(numpy.int).values

        x = data['V_syntc'] / 1023.0
        y0 = data['tau_1'].min()
        y_range = data['tau_1'].max()
        y = data['tau_1'] / y_range

        if len(x.values) < 7:
            return None

        try:
            model, result = self.fit_model(x, y - y0)
        except FloatingPointError:
            return None

        x0, x1 = vsyntc[[0, -1]]
        xi = numpy.arange(int(x0), int(x1) + 1)
        yi = model.eval(params=result.params, x=(xi / 1023.0) + y0) * y_range

        diff = yi[vsyntc - x0] - data['tau_1']
        res = numpy.sqrt(numpy.sum(diff**2)) / len(x)

        return res, xi, yi

    @staticmethod
    def fit_model(x, y, epsilon=1e-6):
        model = models.ExpressionModel('1 /(a * x + b)**d + c')
        model.set_param_hint('a', value=0.1, min=epsilon)
        model.set_param_hint('b', value=1.0, min=epsilon)
        model.set_param_hint('d', value=0.5, min=epsilon)
        params = model.make_params(a=0.1, b=1.0, c=0.0, d=0.5)
        result = model.fit(y, x=x, params=params)
        return model, result

    def make_trafo(self, fit_result):
        if fit_result:
            res, xi, yi = fit_result
            if res < self.residual_tol:
                try:
                    return pycalibtic.Lookup(yi, xi[0])
                except RuntimeError:
                    pass
        return None

    def generate_transformations(self):
        data = self.get_data()
        fits = {}
        errors = {}
        for nrn, nrn_data in data.groupby(level='neuron'):
            nrn_data = nrn_data.dropna()
            nrn_fit = self.approximate_with_fit(nrn_data)
            if nrn_fit is None:
                errors[nrn] = "Fit failed"
            trafo = self.make_trafo(nrn_fit)
            if trafo is None and nrn_fit is not None:
                errors[nrn] = "Trafo failed"
            fits[nrn] = trafo
        return [(self.target_parameter, fits)]


class V_syntci_Calibrator(V_syntc_Calibrator):
    target_parameter = neuron_parameter.V_syntci

    def get_mask(self, data):
        lower_threshold = data['offset'].max() - self.offset_tol
        return ((data['offset'] >= lower_threshold) &
                (data['chi2'] <= self.chi2_tol))


class V_syntcx_Calibrator(V_syntc_Calibrator):
    target_parameter = neuron_parameter.V_syntcx


class I_pl_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.I_pl

    def get_tau0(self, neuron):
        """ Returns the mean of measured tau0s of all experiments
        """
        return np.mean([e.initial_data[neuron]['mean_reset_time'] for e in self.experiments])


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
            tau0 = self.get_tau0(neuron)
            # Extract function coefficients and domain from measured data
            coeffs = self.do_fit(xs, ys, tau0)
            if coeffs != None:
                coeffs = coeffs[::-1] # coeffs are reversed in calibtic transformations
                domain = [tau0, np.nanmax(xs)*1.1] # domain slightly larger than max value
                if self.is_defect(coeffs, domain):
                    transformations[neuron] = None
                else:
                    transformations[neuron] = create_pycalibtic_transformation(coeffs, domain, trafo_type)
            else:
                self.logger.WARN("Fit failed for neuron {}".format(neuron))
                transformations[neuron] = None

        return [(self.target_parameter, transformations)]

    def get_key(self):
        return 'tau_ref'

    def do_fit(self, xs, ys, tau0):
        """ Fits a curve to results of one neuron
            For I_pl, the function is such that the minimum x always returns 1023 DAC
            --> I_pl = 1/(a * x - a * min(x) + 1/1023.)
        """
        def func(x, a):
            return 1/(a*x - a*tau0 + 1/1023.)
        xs = xs[np.isfinite(xs)]
        ys = ys[np.isfinite(xs)]

        ok = self.sanity_check_fit_input(xs, ys)

        if ok == False:
            return None
        else:
            fit_coeffs = curve_fit(func, xs, ys, [0.025e6])[0]
            fit_coeffs = [fit_coeffs[0], 1/1023. - fit_coeffs[0] * tau0]
            return fit_coeffs

    def get_trafo_type(self):
        """ Returns the pycalibtic.transformation type that is used for calibration.
            Default returns Polynomial
        """
        return pycalibtic.OneOverPolynomial

class readout_shift_Calibrator(BaseCalibrator):
    target_parameter = 'readout_shift'

    def __init__(self, experiments, config):
        super(readout_shift_Calibrator, self).__init__(experiments, config)

    def generate_transformations(self):
        """
        """
        results = self.experiments[0].results[0]
        neuron_size = self.experiments[0].measurements[0].sthal.get_neuron_size()
        if neuron_size < 64:
            self.logger.WARN("Neuron size is smaller than 64. 64 is required for readout shift measurement")
        readout_shifts = self.get_readout_shifts(results, neuron_size)
        return [('readout_shift', readout_shifts)]

    def get_neuron_block(self, block_id, size):
        if block_id * size > 512:
            raise ValueError, "There are only {} blocks of size {}".format(512/size, size)
        nids_top = np.arange(size/2*block_id,size/2*block_id+size/2)
        nids_bottom = np.arange(256+size/2*block_id,256+size/2*block_id+size/2)
        nids = np.concatenate([nids_top, nids_bottom])
        neurons = [Coordinate.NeuronOnHICANN(Coordinate.Enum(int(nid))) for nid in nids]
        return neurons

    def get_readout_shifts(self, results, neuron_size):
        """
        """
        trafo_type = self.get_trafo_type()
        n_blocks = 512/neuron_size # no. of interconnected neurons
        readout_shifts = {}
        for block_id in range(n_blocks):
            neurons_in_block = self.get_neuron_block(block_id, neuron_size)
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


class Parrot_Calibrator(BaseCalibrator):
    """
    Filter the neurons that are suitable as parrot neurons
    """

    def __init__(self, experiment, config=None):
        self.experiment = experiment
        backend = config.get_backend_path()
        # TODO more unique?
        self.blacklist = os.path.join(backend, 'parrot_blacklist.txt')
        self.settings = os.path.join(backend, 'parrot_params.pkl')

    def generate_transformations(self):
        """Check which neurons spike via the min-max range of the membrane"""

        # Aggregate data
        data = self.experiment.get_all_data(tuple(), ('v', ))
        amplitudes = data['v'].apply(numpy.ptp).groupby(level='neuron').aggregate([numpy.mean, numpy.std])
        baselines = data['v'].apply(numpy.min).groupby(level='neuron').aggregate([numpy.mean, numpy.std])

        # Mask neurons with to weak PSPs or to large variations in baseline
        V_t = pandas.DataFrame({
            'V_t': baselines['mean'] + amplitudes['mean']/5.0,
            'good': ((amplitudes['mean'] > 0.1) & (baselines['std'] <= 0.015))})

        p = neuron_parameter.V_t
        parrot_calib = {
            'base_parameters' : self.experiment.measurements[0].step_parameters,
            'neuron_parameters' : {n: {p: Volt(V, apply_calibration=True)}
                                   for n, V, ok in V_t.itertuples() if ok},
        }
        parrot_blacklist = [int(n.id()) for n in V_t.index[~V_t['good']]]

        with open(self.settings, 'w') as out:
            cPickle.dump(parrot_calib, out, -1)
        numpy.savetxt(self.blacklist, parrot_blacklist, fmt='%d')
        return []

        threshold = 0.5
        data = self.experiment.get_all_data(tuple(), ('v', ), numeric_index=True)
        ptp = data['v'].groupby(level=['neuron', 'step']).aggregate(
            lambda x: numpy.ptp(x.item()))
        blacklist = (ptp <= threshold).groupby(level='neuron').aggregate(numpy.any)
        neurons = blacklist[blacklist == True].index.get_level_values('neuron')
        with open(self.blacklist, 'w') as blacklist:
            for nrn in neurons:
                blacklist.write('{}\n'.format(nrn))
        with open(self.settings, 'w') as out:
            cPickle.dump(self.experiment.measurements[0].step_parameters, out, -1)
        return []
