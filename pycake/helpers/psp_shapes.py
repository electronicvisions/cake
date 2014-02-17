"""
Definition of PSP shapes that are used by the fitting routine in
fit.fit
"""

import pylab as p
import inspect
from scipy.integrate import trapz


def jacobian(func, p0, epsilon=1e-8):
    """
    Estimate the Jacobian of `func` around `p0`.

    epsilon : float
        dx-value for differential estimate
    """

    def ith_epsilon(i):
        result = p.zeros(len(p0))
        result[i] = epsilon
        return result

    return (p.array([func(p0 + ith_epsilon(i))
                     for i in xrange(len(p0))]) - func(p0)) / epsilon


class PSPShape(object):
    def initial_fit_values(self, time, value):
        """
        Provide an estimate for a parameter fit for the given
        data.

        time : numpy.ndarray
            the time points at which the values were measured

        value : numpy.ndarray
            the measured values
        """
        raise NotImplementedError

    def parameter_limits(self):
        """
        return a dictionary with parameter limits according to
        Minuit standards. E.g., the limit for the parameter x is
        given via dict(limit_x=(min_value, max_value))
        """
        raise NotImplementedError

    def parameter_names(self):
        return inspect.getargspec(self.__call__).args[2:]

    def parameter_dict(self, parameters):
        return dict(zip(self.parameter_names, parameters))

    def __call__(self):
        raise NotImplementedError

    def process_fit_results(self, results, cov_matrix):
        """
        Post-process fit results.

        results : numpy.ndarray
            array of fit parameters

        returns : tuple
            (processed_results, processed_covariance)
        """
        return results, cov_matrix


class DoubleExponentialPSP(PSPShape):
    def __init__(self, min_tau=1e-6, max_tau=1e6):
        """
        min_tau and max_tau are the parameter limits for the synaptic
        time constants
        """
        super(DoubleExponentialPSP, self).__init__()
        self.min_tau = min_tau
        self.max_tau = max_tau

        self.__shape_switch_limit = 1e-8

    @staticmethod
    def __psp_singular(height, times):
        """
        Evaluate the alpha psp for the case tau_1 == tau_2 == 1

        height : float
            The height of the psp at its peak

        times : numpy.ndarray
            array of time points at which the function is evaluated
        """
        A = height * p.exp(1)

        v = A * p.exp(-times) * times
        return v

    @staticmethod
    def __psp_normal(height, tau_frac, t):
        """
        Evaluate the alpha psp for the case tau_1 != tau_2

        tau_frac : float
            ratio tau_1 / tau_2

        t : numpy.ndarray
            array of time points at which the function is evaluated
        """
        A = height / (tau_frac ** (-1. / (tau_frac - 1.))
                      - tau_frac ** (-tau_frac / (tau_frac - 1.)))
        v = A * (p.exp(-t / tau_frac) - p.exp(-t))
        return v

    def __call__(self, time, height, tau_1, tau_2, start, offset):
        """
        evaluate the psp for the given parameters
        """
        tau_1 = p.maximum(tau_1, 0.)
        tau_2 = p.maximum(tau_2, 0.)

        tau_frac = tau_2 / p.float64(tau_1)
        t = p.maximum((time - start) / p.float64(tau_1), 0)

        self.__shape_switch_limit = 1E-8

        if (1. - self.__shape_switch_limit
                < tau_frac
                < 1. + self.__shape_switch_limit):
            return self.__psp_singular(height, t) + offset
        else:
            return self.__psp_normal(height, tau_frac, t) + offset

    def initial_fit_values(self, time, value, smoothing_samples=10,
                           integral_factor=.25, tau_fraction=2):
        """
        Estimate the initial fit values for the given sample data in
        (time, value).

        time : numpy.ndarray
            array of times at which the values are measured

        value : numpy.ndarray
            array of voltage values

        smoothing_samples : int
            width of the box filter used for the convolution

        integral_factor : float
            The time constants are estimated using the integral and
            the height of the given psp. integral_factor is the
            quotient of the maximum of a psp and the integral under
            it, which is 0.25 for tau_fraction = 2 for an ideal psp.

        tau_fraction : float
            The ratio tau_2 / tau_1, which is constant for this
            estimate.
        """
        mean_est_part = int(len(value) * .1)
        mean_estimate = p.mean(value[-mean_est_part:])
        noise_estimate = p.std(value[-mean_est_part:])

        smoothed_value = p.convolve(
            value - mean_estimate,
            p.ones(smoothing_samples) / float(smoothing_samples),
            "same") + mean_estimate

        integral = p.sum(smoothed_value - mean_estimate) * (time[1] - time[0])

        height_estimate = (max(smoothed_value) - mean_estimate)

        min_height = noise_estimate

        if height_estimate < min_height:
            height_estimate = min_height

        t1_est = integral / height_estimate * integral_factor

        # prevent t1 from being smaller than a time step
        t1_est_min = time[1] - time[0]
        if t1_est < t1_est_min:
            t1_est = t1_est_min
        t2_est = tau_fraction * t1_est

        tmax_est = time[p.argmax(smoothed_value)]
        tstart_est = tmax_est + p.log(t2_est / t1_est) \
            * (t1_est * t2_est) / (t1_est - t2_est)

        return dict(
            height=height_estimate,
            tau_1=t1_est,
            tau_2=t2_est,
            start=tstart_est,
            offset=mean_estimate)

    def parameter_limits(self):
        """
        returns a dictionary describing the parameter limits
        """
        return dict(
            limit_tau_1=(self.min_tau, self.max_tau),
            limit_tau_2=(0, self.max_tau))

    def process_fit_results(self, results, covariance_matrix):
        """
        make sure that tau_2 <= tau_1.

        results : numpy.ndarray
            parameters resulting from a parameter fit

        covariance_matrix : numpy.ndarray
            covariance matrix resulting from a parameter fit

        returns : tuple
            a tuple of the new (reordered) result and covariance
            matrix
        """
        processed_r = results.copy()
        if covariance_matrix is None:
            cov_r = None
        else:
            cov_r = covariance_matrix.copy()

        parnames = self.parameter_names()
        tau_1_index = parnames.index("tau_1")
        tau_2_index = parnames.index("tau_2")

        if results[tau_2_index] > results[tau_1_index]:
            processed_r[tau_1_index] = results[tau_2_index]
            processed_r[tau_2_index] = results[tau_1_index]

            orig = [tau_1_index, tau_2_index]
            new = orig[::-1]

            if not cov_r is None:
                cov_r[orig, :] = cov_r[new, :]
                cov_r[:, orig] = cov_r[:, new]

        return processed_r, cov_r

    def __integral(self, time, parameters):
        """
        calculate the `raw` integral

        time : numpy.ndarray
            array of time points for which the integral is being
            calculated

        parameters : numpy.ndarray
            array of parameters for the psp
        """
        par_modified = parameters.copy()

        # ignoring start and offset parameters for this calculation
        par_modified[-1] = 0
        par_modified[-2] = 0

        return trapz(self(time, *par_modified), time)

    def integral(self, time, parameters, covariance_matrix,
                 ):
        """
        calculate the integral for the given
        parameters

        time : numpy.ndarray
            time points at which the function will be evaluated for
            the integral

        parameters : numpy.ndarray
            parameters, for which the PSP integral is evaluated

        covariance_matrix : numpy.ndarray
            covariance matrix for the parameters

        returns : tuple (float, float)
            a tuple of the calculated integral and the estimated error
        """
        j = jacobian(lambda par: self.__integral(time, par),
                     parameters)

        assert covariance_matrix.shape == (len(parameters),
                                           len(parameters))
        assert j[-1] == 0

        # ignoring start and offset parameters for this calculation
        par_modified = parameters.copy()
        par_modified[-1] = 0
        par_modified[-2] = 0
        result = self.__integral(time, par_modified)

        result_error = p.dot(p.dot(j, covariance_matrix), j)

        return result, p.sqrt(result_error)
