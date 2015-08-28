import numpy
import numexpr
from lmfit.models import Model

def psp_singular(height, times):
    """
    Evaluate the alpha psp for the case tau_1 == tau_2 == 1

    height : float
        The height of the psp at its peak

    times : numpy.ndarray
        array of time points at which the function is evaluated
    """
    return height * numpy.exp(1 - times) * times

def psp_normal(height, tau_frac, t):
    """
    Evaluate the alpha psp for the case tau_1 != tau_2

    tau_frac : float
        ratio tau_1 / tau_2

    t : numpy.ndarray
        array of time points at which the function is evaluated
    """
    A = height / (tau_frac ** (-1. / (tau_frac - 1.))
                  - tau_frac ** (-tau_frac / (tau_frac - 1.)))
    # A little bit faster with numexpr
    return numexpr.evaluate("A * (exp(-t / tau_frac) - exp(-t))")

def double_exponetial_psp(time, height, tau_1, tau_2, start, offset):
    """
    evaluate the psp for the given parameters
    """
    tau_1 = numpy.maximum(tau_1, 0.)
    tau_2 = numpy.maximum(tau_2, 0.)

    if tau_1 == tau_2:
        t = numpy.maximum((time - start) / numpy.float64(tau_1), 0)
        return psp_singular(height, t) + offset
    elif tau_1 == 0.0:
        return time * numpy.nan
    else:
        t = numpy.maximum((time - start) / numpy.float64(tau_1), 0)
        tau_frac = numpy.float64(tau_2) / numpy.float64(tau_1)
        return psp_normal(height, tau_frac, t) + offset


class DoubleExponentialPSPModel(Model):
    """
    Attributes:
        smoothing_samples : int
            Parameter for guess(): width of the box filter used for
            the convolution

        integral_factor : float
            Parameter for guess(): The time constants are estimated
            using the integral and the height of the given
            psp. integral_factor is the quotient of the maximum of
            a psp and the integral under it, which is 0.25 for
            tau_fraction = 2 for an ideal psp.

        tau_fraction : float
            Parameter for guess(): The ratio tau_2 / tau_1, which is
            constant for this estimate.
    """
    def __init__(self, func=None, *args, **kwargs):
        self.smoothing_samples = 10
        self.integral_factor = 0.25
        self.tau_fraction = 2.0
        if func is None:
            func = double_exponetial_psp
        super(DoubleExponentialPSPModel, self).__init__(func, *args, **kwargs)

    def make_params(self, **kwargs):
        params = super(DoubleExponentialPSPModel, self).make_params(**kwargs)
        params['tau_1'].set(min = 0.0)
        params['tau_2'].set(min = 0.0)
        return params

    def _residual(self, params, data, weights, **kwargs):
        """Overwrite default residuals to get an correct for the chi2
        See: http://stackoverflow.com/questions/15255928
        """
        # Do this to include errors as weights.
        diff = self.eval(params, **kwargs) - data
        if weights is not None:
            return numpy.asarray(numpy.sqrt(diff ** 2 / weights ** 2))
        else:
            return numpy.asarray(diff)  # for compatibility with pandas.Series

    def guess(self, data, time=None, **kwargs):
        """
        Estimate the initial fit values for the given sample data in
        (time, value). This function uses the class attributes
        smoothing_samples, integral_factor and tau_fration as paramters
        for the guess.

        Parameters:
            time : numpy.ndarray
                array of times at which the values are measured

            data : numpy.ndarray
                array of voltage values
        """
        mean_est_part = int(len(data) * .1)
        mean_estimate = numpy.mean(data[-mean_est_part:])
        noise_estimate = numpy.std(data[-mean_est_part:])

        smoothed_data = numpy.convolve(
            data - mean_estimate,
            numpy.ones(self.smoothing_samples) / float(self.smoothing_samples),
            "same") + mean_estimate

        integral = numpy.trapz(smoothed_data - mean_estimate, time)

        height_estimate = (max(smoothed_data) - mean_estimate)

        min_height = noise_estimate

        if height_estimate < min_height:
            height_estimate = min_height

        t1_est = integral / height_estimate * self.integral_factor

        # prevent t1 from being smaller than a time step
        t1_est_min = time[1] - time[0]
        if t1_est < t1_est_min:
            t1_est = t1_est_min
        t2_est = self.tau_fraction * t1_est

        tmax_est = time[numpy.argmax(smoothed_data)]
        tstart_est = tmax_est + numpy.log(t2_est / t1_est) \
            * (t1_est * t2_est) / (t1_est - t2_est)

        return self.make_params(
            height=height_estimate,
            tau_1=t1_est,
            tau_2=t2_est,
            start=tstart_est,
            offset=mean_estimate)

def fit(t, v, error_estimate):
    model = DoubleExponentialPSPModel()
    params = model.guess(v, time=t)
    result = model.fit(v, time=t, weights=error_estimate, params=params)
    return result
