import utils

import pylab as p
import scipy.signal

import pickle

# TODO: use official constants
# slow hicann clock (clock of the current pulse generator)
slow_clock = 25.0e6
# adc clock (sampling interval of voltage data)
adc_clock = 96.0e6

# TODO: move to appropriate module
def memoize(f):
    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret
    return memodict().__getitem__

class TaumExperiment(object):
    def __init__(self, voltage, time, pulselength):
        self.voltage = voltage
        self.time = time
        self.pulselength = pulselength

class TaumSweepAnalysis(object):
    def __init__(self, experiment, clock_shift_correction=None):
        self.experiment = experiment

        self.stimulus_period_length = (
            adc_clock / slow_clock
            * (self.experiment.pulselength + 1)
            * 129.)

        if clock_shift_correction != None:
            self._corrected_period = (
                self.stimulus_period_length
                * (1. + clock_shift_correction))
        else:
            self._corrected_period = None

    def time_axis(self, length):
        """
        Returns the time axis for self.experiment.voltage (unit is seconds)
        """
        return p.arange(length) * 1.0 / adc_clock

    def plot_voltage(self, cutoff=None):
        if cutoff is None:
            time = self.time_axis(len(self.experiment.voltage))
            p.plot(time * 1e6, self.experiment.voltage)
        else:
            time = self.time_axis(cutoff)
            p.plot(time * 1e6, self.experiment.voltage[:cutoff])

        p.xlabel("time [us]")
        p.ylabel("membrane voltage [V]")

    def plot_autocorr(self):
        v = self.experiment.voltage
        v -= p.mean(v)
        corr1 = p.convolve(v[5:], v[::-1], "valid")
        p.plot(corr1)
        p.xlabel("index [sampling interval]")
        p.ylabel("autocorr. [V^2]")

    # TODO: memoize this
    def circular_2nd_derivative_filtered(self, data, filter_width, dt):
        """
        Estimate the 2nd circular derivative of `data` and return
        the filtered result.

        dt is the sampling time of data

        The filter kernel is a Gaussian with a width of `filter_width`
        in the same unit as dt.
        """
        diff = p.roll(data, 1) - data
        diff2 = diff - p.roll(diff, -1)
        diff2 /= (dt ** 2)

        kernel = scipy.signal.gaussian(len(data), filter_width / dt)
        kernel /= p.sum(kernel)
        kernel = p.roll(kernel, int(len(data) / 2))

        return p.real(p.ifft(p.fft(kernel) * p.fft(diff2)))

    def calc_corrected_period(self, factor_range=1e-4):
        """
        Calculate the optimal re-occurrence period of the voltage
        sample using utils.tune_period.

        factor_range: float
            the maximal deviation factor that is considered in both
            directions

        Returns a tuple of
            (the index of the optimal correction factor,
            array of all considered factors,
            the corrected period in samples (floating point)
            )
        """
        optimum, factors, stds = utils.tune_period(
            self.experiment.voltage,
            self.stimulus_period_length,
            [-factor_range, factor_range])

        corrected_period = self.stimulus_period_length * (1. + factors[optimum])
        return optimum, factors, stds, corrected_period

    def get_optimal_period(self):
        """
        Return the optimal period as calculated by
        calc_corrected_period.
        """
        if self._corrected_period is None:
            _, _, _, self._corrected_period = self.calc_corrected_period()
        return self._corrected_period

    def plot_period_optimization(self):
        optimum, factors, stds, corrected_period = self.calc_corrected_period()

        ppms = p.array(factors) * 1e6
        p.plot(ppms, stds, 'x', label="std")
        p.xlabel("clock offset [ppm]")
        p.ylabel("mean standard deviation of signal [V]")
        p.axvline(ppms[optimum])
        p.legend()

    # TODO: memoize this
    def calc_averaged_voltage(self, offset=0):
        """
        Calculate the average voltage trace over all iterations
        of the repeating signal.

        offset: int
            number of samples that are truncated at the beginning
            of the voltage recording

        Returns a tuple of
        (The mean voltage (array),
        standard deviation of the voltage over all samples (array),
        the number of samples over which the average was taken
        )
        """
        reshaped = utils.nonint_reshape_data(
            self.experiment.voltage[offset:],
            self.get_optimal_period())

        mean_v = p.mean(reshaped, axis=0)
        std_v = p.std(reshaped, axis=0, ddof=1)
        repetition_count = reshaped.shape[0]

        return mean_v, std_v, repetition_count

    def plot_averaged_voltage(self, offset=0):
        mean_v, std_v, rep_num = self.calc_averaged_voltage(offset)
        time = self.time_axis(len(mean_v))

        p.plot(time * 1e6, mean_v, color="k", lw=2,
               label="mean of {} repetitions".format(rep_num))
        p.fill_between(time * 1e6, mean_v - std_v, mean_v + std_v, alpha=.5)

        p.plot(
            time * 1e6,
            self.experiment.voltage[offset:offset + len(mean_v)],
            alpha=.5,
            color="red",
            label="single trace")

        p.legend()
        p.xlabel("time [us]")
        p.ylabel("membrane voltage [V]")

    def calc_zero_current_offset(self, filter_width=100e-9):
        mean_v, _, _ = self.calc_averaged_voltage(self.get_optimal_period())
        diff2 = self.circular_2nd_derivative_filtered(
            mean_v,
            filter_width,
            1.0 / adc_clock)
        zero_current_start = p.argmin(diff2)
        return zero_current_start

    def plot_2nd_derivative(self, filter_width=100e-9):
        mean_v, _, _ = self.calc_averaged_voltage(self.get_optimal_period())
        diff2 = self.circular_2nd_derivative_filtered(
            mean_v,
            filter_width,
            1.0 / adc_clock)

        time = self.time_axis(len(diff2))
        p.plot(
            time * 1e6,
            diff2,
            label="second deriv., smooth={0} us".format(filter_width * 1e6))
        p.legend()
        p.xlabel("time [us]")
        p.ylabel("d^2V/dt^2 [V/s^2]")

        offset = self.calc_zero_current_offset()
        p.axvline(time[offset] * 1e6)

    def get_decay_fit_range(self, signal_length):
        zero_length = (1. - self.experiment.stim_length / 129.)
        fitstart = int(0.01 * signal_length)
        fitstop = int(zero_length * signal_length * 0.99)
        fittime = p.arange(fitstart, fitstop)
        return fitstart, fitstop, fittime

    def fit_exponential(self):
        func = lambda x, tau, offset, a: a * p.exp(-(x - x[0]) / tau) + offset

        mean_v, std_v, rep_num = self.calc_averaged_voltage(
            self.calc_zero_current_offset())

        fitstart, fitstop, fittime = self.get_decay_fit_range(len(mean_v))

        error_of_mean = std_v[fitstart:fitstop] / p.sqrt(rep_num)

        from scipy.optimize import curve_fit

        expf, pcov, infodict, errmsg, ier = curve_fit(
            func,
            fittime,
            mean_v[fitstart:fitstop],
            [.5, 100., 0.1],
            sigma=error_of_mean,
            full_output=True
            )
        print expf, pcov, infodict

        tau = expf[0] / adc_clock * 1e6
        #tau_sigma = abs(p.sqrt(pcov[0, 0])) / adc_clock * 1e6
        #print "tau =", tau, "+/-", tau_sigma, "us"

        #DOF = len(fittime) - len(expf)
        #red_chisquared = sum(infodict["fvec"] ** 2) / (DOF)
        #print "Reduced Chi^2", red_chisquared

        #residuals = infodict["fvec"]

        return tau

    def plot_fit_residuals(self, sp1, sp2):
        expf, func, fittime, fitdata, pcov, infodict, ier, red_chisquared = self.fit_exponential()

        time = self.time_axis(len(infodict["fvec"]))

        p.subplot(sp1)
        p.plot(time * 1e6, fitdata, 'rx', label="data")
        p.plot(time * 1e6, func(fittime, *expf), color="black", label="fit")
        p.xlabel("time [us]")
        p.ylabel("membrane voltage [V]")

        p.subplot(sp2)
        p.errorbar(time * 1e6, infodict["fvec"], yerr=1)
        p.xlabel("time [us]")
        p.ylabel("reduced residuals")

        p.axhline(0)

    def calc_diff_approximations(self, length, v_err_est=None):
        """
        length:
        """
        mean_v, std_v, num_means = self.calc_averaged_voltage(
            self.calc_zero_current_offset())

        fitstart, fitstop, fittime = self.get_decay_fit_range(len(mean_v))

        xvals = p.arange(length) - length * .5
        xvalsquaresum = sum(xvals ** 2)

        slope_means = []
        slope_errors = []
        voltage_means = []
        voltage_errors = []
        chi2_values = []

        for fs in p.arange(fitstart, fitstop - length, int(length / 2)):
            data = mean_v[fs:fs + length]
            errs = std_v[fs:fs + length] / p.sqrt(num_means)

            r1 = p.mean(data)
            r1_err = p.sqrt(sum((data * errs) ** 2) / len(data))

            r0 = sum(xvals * (data - r1)) / xvalsquaresum

            if v_err_est == None:
                membrane_error_estimate = errs
            else:
                membrane_error_estimate = v_err_est

            r0_err = p.sqrt(sum((xvals ** 2)
                                * (membrane_error_estimate ** 2))) \
                     / sum(xvals ** 2)

            slope = r0 * adc_clock
            slope_err = r0_err * adc_clock
            print "slope:", slope, "+/-", slope_err
            res = ((data - (xvals * r0 + r1)) / errs) ** 2
            fit_chi2 = sum(res) / (length - 2)
            print "Fit reduced Chi2:", fit_chi2

            slope_means.append(slope)
            slope_errors.append(slope_err)
            voltage_means.append(r1)
            voltage_errors.append(r1_err)
            chi2_values.append(fit_chi2)

        return (slope_means,
                slope_errors,
                voltage_means,
                voltage_errors,
                chi2_values)

    def plot_diff_approximations(self, length):
        slope_means, slope_errors, voltage_means, voltage_errors, \
                     chi2_values = self.calc_diff_approximations(length)
        p.errorbar(voltage_means, slope_means,
                   xerr=voltage_errors, yerr=slope_errors,
                   label="estimated slope (data)",
                   fmt="kx")

        poly = p.polyfit(voltage_means, slope_means, 1)
        p.plot(voltage_means, p.poly1d(poly)(voltage_means), label="fit")

        print "dV/dt / V:", poly[0], "1/s", "inverse:", 1. / poly[0] * 1e6, "us"

        p.xlabel("membrane voltage [V]")
        p.ylabel("dV/dt [V/s]")
        p.legend()

