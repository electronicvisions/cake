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
from pyhalco_common import Enum
import pyhalco_hicann_v2 as Coordinate
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
    targeted_defect_coordinate = None

    def __init__(self, experiment, config):
        self.experiment = experiment

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

    def get_results(self, keys=None):
        """
        get all data for the columns with keywords 'keys' in [DataFrame]
        experiment.results. If key is None, return the keys specified by the
        class attributes target_parameter and result_key
        """
        results = []
        if keys is None:
            keys = [self.target_parameter, self.get_key()]
        return self.experiment.get_all_data(keys)

    def generate_transformations(self):
        """ Takes averaged experiments and does the fits
        """
        transformations = {}
        trafo_type = self.get_trafo_type()
        for neuron, group in self.get_results().dropna(how='any').groupby(level='neuron'):
            # Need to switch y and x in order to get the right fit
            # (y-axis: configured parameter, x-axis: measurement)
            ys_raw, xs_raw = group.T.values
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
        return self.experiment.measurements[0].get_neurons()

    def get_trafo_type(self):
        """ Returns the pycalibtic.transformation type that is used for calibration.
            Default returns Polynomial
        """
        return pycalibtic.Polynomial


class V_reset_Calibrator(BaseCalibrator):
    target_parameter = shared_parameter.V_reset

    def get_key(self):
        return 'baseline'

    def generate_transformations(self):
        """
        calculates the mean baseline over the FGBlocks and fits a line on the
        data for each FGBlock.

        Returns:
            List of tuples, each containing the neuron parameter and a
            dictionary containing polynomial fit coefficients for each neuron
        """
        results = self.get_results().dropna(how='any')
        results_per_FGBlock = results.groupby(level='shared_block')
        trafos = {}
        trafo_type = self.get_trafo_type()
        for fgblock, res in results_per_FGBlock:
            mean = res.groupby('V_reset').mean()
            xs = self.prepare_x(mean.index)
            ys = self.prepare_y(mean.values[:,0])
            coeffs = list(self.do_fit(ys, xs))
            coeffs = coeffs[::-1] # reverse coefficients for calibtic
            domain = self.get_domain(ys)
            trafo = create_pycalibtic_transformation(coeffs, domain, trafo_type)
            trafos[fgblock] = trafo
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
            fit_coeffs = [fit_coeffs[1], fit_coeffs[0], 0]
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
    v_range=0.150

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
        raise NotImplementedError(self)

    def find_optimum(self, data, v_range=0.150, spiking_threshold=0.001):

        fit_data = self.prepare_data(data.copy(), v_range, spiking_threshold)
        if len(fit_data[0]) < 4:
            self.logger.WARN("V_convoff_Calibrator: Not enough valid data points: {}. "
                             "Expected at least 4. Spiking threshold {}".format(len(fit_data[0]),
                                                                                spiking_threshold))
            return fit_data, None, None

        results = []
        params = Parameters()
        params.add('a', value=-1.0)
        params.add('b', value=fit_data[0][len(fit_data)/2])
        params.add('c', value=0.0)

        out = minimize(self.residual, params,
                       args=(fit_data[0], fit_data[1], 1.0))
        return fit_data, numpy.sum(out.residual**2), out

    def plt_fits(self, axis, results):
        fit_data, res, fit_result = results
        x, y = fit_data
        axis.plot(x, y, color='k', label="measured")
        #axis.plot(
        #    x, f(params, x), 'x', 
        #    label='Res**2: {:.2f}, b: {:.2f}, chi2: {:.5f}'.format(
        #        res, params['b'].value, out.chisqr))
        if fit_result is not None:
            params = fit_result.params
            x = numpy.linspace(0, 1.0, 100)
            l = axis.plot(x, self.f(params, x), '--',
                          label='fitted, '
                                'a: {:.2f}, '
                                'b: {:.2f}, '
                                'c: {:.2f}'.format(params['a'].value,
                                                   params['b'].value,
                                                   params['c'].value))
            axis.plot([params['b'].value], [params['c'].value], 'x',
                      color=l[0].get_color())
        axis.set_xlabel("V_convoff [DAC] / 1023")
        axis.set_ylabel("shift / {:.2f} V".format(self.v_range))

    def plt_residuals(self, axis, results):
        axis.plot([res for res, _, _ in results])

    @staticmethod
    def V_convoff(results):
        _, _, fit_result = results
        if fit_result is not None:
            params = fit_result.params
            return params['b'].value * 1023

    def plot_fit_for_neuron(self, nrn, axis, v_range=None):
        if v_range is None:
            v_range = self.v_range
        data = self.get_results([self.target_parameter, 'mean', 'std'])
        results = self.find_optimum(
            data, v_range, self.get_spiking_threshold(nrn))
        self.plt_fits(axis, results)
        return results

    def get_spiking_threshold(self, nrn):
        """
        Get the the threshold for considering a neuron spiking
        """
        return self.experiment.initial_data[nrn].get('std', 0.001) * 1.5

    def believably(self, value):
        """
        Returns True if the value of V_convoffi/x is within a believably range
        """
        return 100 < value < 900

    def generate_transformations(self):
        fits = {}
        data = self.get_results([self.target_parameter, 'mean', 'std']).dropna(how='any')
        for nrn, data in data.groupby(level='neuron'):
            results = self.find_optimum(
                data, self.v_range, self.get_spiking_threshold(nrn))
            value = self.V_convoff(results)
            if self.believably(value):
                fits[nrn] = Constant(value)
            elif value is not None:
                self.logger.WARN("V_convoff_Calibrator: Unbelievable value {}.".format(value))
                fits[nrn] = None
            else:
                fits[nrn] = None
        return [(self.target_parameter, fits)]


class V_convoffi_Calibrator(V_convoff_Calibrator):
    target_parameter = neuron_parameter.V_convoffi
    v_range=0.150

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

    def __init__(self, experiment, config=None, calib_feature='tau_1'):
        self.experiment = experiment
        self.offset_tol = 0.025
        self.chi2_tol = 30
        self.residual_tol = 0.05
        # The signal-to-noise ratio ensures that sufficiently strong PSPs
        # were detected to provide a good fit, otherwise the extracted
        # time-constants are not meaningful. If a full half of neurons is
        # blacklisted by this criteria, most likely the synapse driver used on
        # this side is faulty
        self.signal_to_noise_tol = 2.0
        self.calib_feature = calib_feature
        self.fits = {}  # Cache for fit results
        self.calibrations = {}  # Cache for calibration

        #  Softplus function to fit to the logarithm of the time constants
        self.model = models.ExpressionModel(
            'a * log(1 + exp(c * (b - x))) / c + offset')
        self.model.set_param_hint('a', value=0.3)
        self.model.set_param_hint('b', value=0)
        self.model.set_param_hint('c', value=0.01)
        self.model.set_param_hint('offset', value=-16)

    def get_mask(self, data):
        """
        Applies the limits given by self.offset_tol, self.chi2_tol and
        self.signal_to_noise_tol to the data

        Arguments:
            data: pandas.DataFrame as obtained by self.get_data()

        Returns:
            pandas.Series with boolean flags, True marks good data points
        """
        def threshold(data):
            "Find values that are to high"
            return data['offset'] < (data['offset'].min() + self.offset_tol)

        return (
            (data.groupby(level='neuron', group_keys=False).apply(threshold)) &
            (data['chi2'] <= self.chi2_tol) &
            (data['signal_to_noise'] >= self.signal_to_noise_tol))

    def get_data(self):
        """
        Retrieve data for the fit. Adds x, y and mask columns for the fit

        Returns:
            pandas.DataFrame
        """
        data = self.get_results([self.target_parameter,
                    self.calib_feature, 'start', 'offset', 'chi2',
                    'signal_to_noise', 'mean', 'std'])
        data['x'] = data[self.target_parameter.name]
        data['y'] = data[self.calib_feature]
        data['mask'] = self.get_mask(data)
        return data

    def approximate_with_fit(self, nrn, nrn_data):
        data = nrn_data[nrn_data['mask']].dropna()

        if len(data) < 6:
            self.logger.WARN(
                "Fit for {} failed: Not enough valid data points: {}".format(nrn, len(data)))
            return

        x = data['x'].values
        y = numpy.log(data['y']).values

        p = self.model.make_params()
        p['offset'].set(value=y.min())

        try:
            result = self.model.fit(y, p, x=x)
        except (FloatingPointError, ValueError) as e:
            self.logger.WARN(
                "Fit for {} failed: {}".format(nrn, e))
            return None

        if not result.success:
            self.logger.WARN(
                "Fit for {} failed: {}".format(nrn, result.message))
            return None

        # Exp creates too large numbers for some fits, replace with NaNs and drop
        # them.
        with numpy.errstate(over='ignore'):
            xi = numpy.arange(1024)
            fitted = pandas.Series(
                    numpy.exp(self.model.eval(result.params, x=xi)), index=xi)
            fitted.replace([np.inf, -np.inf], np.nan, inplace=True)
            fitted.dropna(inplace=True)

        # The domain encodes the value range for which we have valid input data
        domain = [fitted.loc[x.max()], fitted.loc[x.min()]]
        if domain[0] >= domain[1]:
            self.logger.WARN("For neuron {}: We expect that the measured values descend when "
                             "increasing the parameter value, but the measured value for the "
                             "highest parameter value is greater or equal than that for the "
                             "lowest. Neuron is blacklisted.".format(nrn))
            return None

        return fitted, domain, result

    def make_trafo(self, nrn, fit_result):
        if fit_result is None:
            return None
        fitted, domain, result = fit_result
        if result.redchi >= self.residual_tol:
            self.logger.WARN(
                "Failed to create Lookup transformation for {}: residuals to large: {}".format(nrn, result.redchi))
            return None

        # `pycalibtic.Lookup` requires consecutive x values.
        # This could be a problem due to `fitted.dropna()` above.
        if not numpy.all(numpy.diff(fitted.index.values) == 1):
            wrong = numpy.nonzero(numpy.diff(fitted.index.values) != 1)[0]
            # provide some context
            wrong = numpy.sort(numpy.concatenate((wrong - 1, wrong, wrong + 1)))
            wrong = wrong[wrong >= 0]
            self.logger.WARN(
                "Failed to create Lookup transformation for {}:\n"
                "non-consecutive x values after fit:\n".format(
                    nrn, fitted.iloc[wrong]))
            return None

        try:
            lookup = pycalibtic.Lookup(fitted.values, fitted.index.values[0])
            lookup.setDomain(*domain)
        except RuntimeError as e:
            self.logger.WARN(
                "Failed to create Lookup transformation for {}: {}".format(nrn, e))
            return None

        return lookup

    def generate_transformations(self):
        data = self.get_data()
        for nrn, nrn_data in data.groupby(level='neuron'):
            self.fits[nrn] = self.approximate_with_fit(nrn, nrn_data)
            self.calibrations[nrn] = self.make_trafo(nrn, self.fits[nrn])
        return [(self.target_parameter, self.calibrations)]


class V_syntci_Calibrator(V_syntc_Calibrator):
    target_parameter = neuron_parameter.V_syntci

    def get_mask(self, data):
        def threshold(data):
            "Find values that are to high"
            return data['offset'] >= (data['offset'].max() - self.offset_tol)

        return (
            (data.groupby(level='neuron', group_keys=False).apply(threshold)) &
            (data['chi2'] <= self.chi2_tol) &
            (data['signal_to_noise'] >= self.signal_to_noise_tol))


class V_syntcx_Calibrator(V_syntc_Calibrator):
    target_parameter = neuron_parameter.V_syntcx

class I_gl_PSP_Calibrator(V_syntc_Calibrator):

    target_parameter = neuron_parameter.I_gl

    def __init__(self, experiment, config=None):
        super(I_gl_PSP_Calibrator, self).__init__(experiment, config, calib_feature='tau_2')
        self.offset_tol = 1
        self.residual_tol = 0.05

    def approximate_with_fit(self, nrn, nrn_data):
        tmp = V_syntc_Calibrator.approximate_with_fit(self, nrn, nrn_data)
        if tmp is not None:
            fitted, domain, result = tmp
            # TODO: domain: set save minimum value for I_gl
            return fitted, domain, result
        else:
            return None


class I_pl_Calibrator(BaseCalibrator):
    target_parameter = neuron_parameter.I_pl


    def generate_transformations(self):
        """ Takes averaged experiments and does the fits
        """
        transformations = {}
        trafo_type = self.get_trafo_type()
        results = self.get_results().dropna(how='any').groupby(level='neuron')

        for neuron, group in results:
            # Need to switch y and x in order to get the right fit
            # (y-axis: configured parameter, x-axis: measurement)
            ys_raw, xs_raw = group.T.values
            xs = self.prepare_x(xs_raw)
            ys = self.prepare_y(ys_raw)
            # Extract function coefficients from measured data
            coeffs = self.do_fit(xs, ys)

            if coeffs is None:
                self.logger.WARN("Fit failed for neuron {}".format(neuron))
                transformations[neuron] = None
            else:
                coeffs = coeffs[::-1] #reverse order of coeffs bc. calibtic transformation has reverse order
                #lower bound of domain is singular point of the fit function
                x_0 = -coeffs[0]/coeffs[1]
                lower_bound = x_0+1e-11 if x_0 > 0 else 0
                #lower bound slightly larger than singular point. Upper bound is not really determined and
                #thus set to a large value
                domain = [lower_bound, 1e-4]
                if self.is_defect(coeffs, domain):
                    transformations[neuron] = None
                else:
                    transformations[neuron] = create_pycalibtic_transformation(coeffs, domain, trafo_type)

        return [(self.target_parameter, transformations)]

    def get_key(self):
        return 'tau_ref'

    def do_fit(self, xs, ys):
        """ Fits a curve to results of one neuron
            I_pl = 1/(a*x + b)
        """
        def func(x, a, b):
            return 1./(a*x + b)
        xs = xs[np.isfinite(xs)]
        ys = ys[np.isfinite(xs)]

        ok = self.sanity_check_fit_input(xs, ys)

        if ok == False:
            return None
        else:
            #starting values are taken from Schwartz ideal transformation
            fit_coeffs = curve_fit(func, xs, ys, p0=[6.25e6/1023., 1/1023.])[0]
            return fit_coeffs

    def get_trafo_type(self):
        """ Returns the pycalibtic.transformation type that is used for calibration.
            Default returns Polynomial
        """
        return pycalibtic.OneOverPolynomial

class readout_shift_Calibrator(BaseCalibrator):
    """
    Determine per-denmem readout buffer offset from average interconnected resting potential
    """
    target_parameter = 'readout_shift'

    def generate_transformations(self):
        """
        """
        for measurement in self.experiment.measurements:
            neuron_size = measurement.sthal.get_neuron_size()
            if neuron_size != 64:
                raise ValueError("Neuron size is smaller than 64. 64 is required for readout shift measurement")

        readout_shifts = self.get_readout_shifts(neuron_size)
        return [(self.target_parameter, readout_shifts)]

    def get_neuron_block(self, block_id, size):
        # FIXME replace by LogicalNeuron coordinate
        # from pymarocco_coordinates import LogicalNeuron
        # will be part of halco later
        if block_id * size > 512:
            raise ValueError, "There are only {} blocks of size {}".format(512/size, size)
        nids_top = np.arange(size/2*block_id,size/2*block_id+size/2)
        nids_bottom = np.arange(256+size/2*block_id,256+size/2*block_id+size/2)
        nids = np.concatenate([nids_top, nids_bottom])
        neurons = [Coordinate.NeuronOnHICANN(Enum(int(nid))) for nid in nids]
        return neurons

    def get_readout_shifts(self, neuron_size):
        """
        calculate the readout shift as: 1. average over all measurements for
        all neurons (i.e. get a single number from all measurements). 2. average for
        every neuron over the number of repetitions. 3. subtract the mean over all
        neurons for every single neuron
        create the calibtic transformation.

        Args:
            neuron_size: number of interconnected denmems
        returns:
            readout_shifts: [dict] readout shifts for all neurons
        """
        trafo_type = self.get_trafo_type()
        n_blocks = 512/neuron_size # no. of interconnected neurons
        readout_shifts = {}

        results = self.get_results([self.get_key()]).dropna(how='any')
        for block_id in range(n_blocks):
            neurons_in_block = self.get_neuron_block(block_id, neuron_size)

            results_block = results.loc[neurons_in_block]
            shifts = results_block.mean(level='neuron') - results_block.mean()
            for i, shift in enumerate(shifts.values):
                trafo = create_pycalibtic_transformation(list(shift), None, trafo_type)
                readout_shifts[shifts.index[i]] = trafo
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
        parrot_blacklist = [int(n.toEnum()) for n in V_t.index[~V_t['good']]]

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
