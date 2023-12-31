#!/usr/bin/env python

import matplotlib
matplotlib.use("agg")

import unittest
import pylab as p
import numpy
from numpy.random import seed
from pycake.helpers.psp_shapes import DoubleExponentialPSP
from cake_test_psp_shapes import noisy_psp
from pycake.helpers.psp_fit import fit
from pycake.helpers import psp_fit
import mock


class TestFit(unittest.TestCase):
    def calculate_fit_quality_fixed_seed(self, seed_val, debug_plot=False,
                                         max_dev=4.):
        """
        Assert the quality of the fit by comparing the parameters used
        to generate test data to the fit result. The deviation should
        be smaller than `max_dev` times the error estimate reported by
        the fit routine. (e.g., using max_dev=3 leads to a statistical
        failure probability of 0.3% assuming a normal distribution on
        the results.)

        Only the height and tau_1/tau_2 are tested for a correct error
        estimate.
        """

        noise = .1

        seed(seed_val)
        times = p.arange(0, 100, .1)

        height = 1.
        tau_1 = 10.
        tau_2 = 5.
        start = 30.
        offset = 50.

        voltage = noisy_psp(height, tau_1, tau_2, start, offset, times, noise)

        success, fitres, cov, red_chi2 = fit(
            DoubleExponentialPSP(),
            times,
            voltage,
            noise,
            fail_on_negative_cov=[True, True, True, False, False])

        if debug_plot:
            p.figure()
            p.plot(times, DoubleExponentialPSP()(times, *fitres), 'r-')
            p.errorbar(times, voltage, yerr=noise, fmt='bx')
            p.xlabel("time / AU")
            p.ylabel("voltage / AU")
            p.title("fit result")
            fname = "/tmp/fit_quality_plot_{0}.pdf".format(seed_val)
            p.savefig(fname)
            print("Plot saved to:", fname)

        err = p.sqrt(p.diag(cov))

        # print "seed:", seed_val

        self.assertTrue(success)
        self.assertLess(abs(fitres[0] - height),
                        max_dev * err[0])
        self.assertLess(abs(fitres[1] - tau_1), max_dev * err[1])
        self.assertLess(abs(fitres[2] - tau_2), max_dev * err[2])

        # NOTE: only testing height and time constants for correct
        #       error estimate
        # self.assertLess(abs(fitres[3] - start), max_dev * err[3])
        # self.assertLess(abs(fitres[4] - offset),
        #                 max_dev * err[4])

        self.assertLess(red_chi2, 1.5)
        if debug_plot:
            print(red_chi2, abs(fitres[1] - tau_1) / err[1], abs(fitres[2] - tau_2) / err[2])

    def test_fit_quality_fixed_seeds(self):
        """
        Test the fit quality of an DoubleExponentialPSP at certain (fixed) seeds.
        """
        self.calculate_fit_quality_fixed_seed(31205)
        self.calculate_fit_quality_fixed_seed(3120945)

        for seed_o in range(50):
            self.calculate_fit_quality_fixed_seed(3901225 + seed_o)

    # This test regularly fails (once to twice a week in Jenkins),
    # wasting MK's time looking at failures. Therefore disabled.
    @unittest.skip("failing too often")
    def test_fit_quality_random_seeds(self):
        """
        Test the fit quality with random seeds. Note: this can
        fail indeterministically (as suggested by CK and ECM).
        """

        for _ in range(50):
            self.calculate_fit_quality_fixed_seed(None)

    def test_param_estimate(self, debug_plot=False):
        """
        Test the parameter estimate for Alpha-shaped PSPs.
        (Prevention of code accidents)
        """
        times = p.arange(0, 100, .1)

        height = 4.
        tau_1 = 20.
        tau_2 = 1.
        start = 30.
        offset = 50.
        noise = .0001

        voltage = noisy_psp(height, tau_1, tau_2, start, offset, times, noise)

        params = DoubleExponentialPSP().initial_fit_values(times, voltage)

        self.assertAlmostEqual(params["start"], 26.69, 2)
        self.assertAlmostEqual(params["tau_1"], 4.6956, 2)
        self.assertAlmostEqual(params["tau_2"], 9.39, 2)
        self.assertAlmostEqual(params["offset"], 50.19, 2)
        self.assertAlmostEqual(params["height"], 3.798, 2)

        if debug_plot:
            p.figure()
            p.plot(times, DoubleExponentialPSP()(times, **params), 'r-')
            p.errorbar(times, voltage, yerr=noise, fmt='bx')
            p.xlabel("time / AU")
            p.ylabel("voltage / AU")
            p.title("initial parameter estimate")
            fname = "/tmp/test_param_estimate.pdf"
            p.savefig(fname)
            print("plot saved to:", fname)


    def test_errors(self):
        """
        Test behavior of fit in various error cases
        """

        noise = .1

        times = p.arange(0, 100, .1)

        height = 1.
        tau_1 = 10.
        tau_2 = 5.
        start = 30.
        offset = 50.

        voltage = noisy_psp(height, tau_1, tau_2, start, offset, times, noise)

        with mock.patch('pycake.helpers.psp_fit.optimize.leastsq', side_effect=FloatingPointError):
            success, fitres, cov, red_chi2 = fit(
                DoubleExponentialPSP(),
                times,
                voltage,
                noise,
                fail_on_negative_cov=[True, True, True, False, False])

            self.assertEqual(success, False)
            self.assertEqual(fitres, None)
            self.assertEqual(cov, None)
            self.assertEqual(red_chi2, None)


        with mock.patch('pycake.helpers.psp_fit.optimize.leastsq', side_effect=RuntimeError):
            success, fitres, cov, red_chi2 = fit(
                DoubleExponentialPSP(),
                times,
                voltage,
                noise,
                fail_on_negative_cov=[True, True, True, False, False])

            self.assertEqual(success, False)
            self.assertEqual(fitres, None)
            self.assertEqual(cov, None)
            self.assertEqual(red_chi2, None)


        fake_ret = [
            numpy.array([31., 5., 10., 51., 2.]),
            numpy.ones((5, 5)),
            "",
            "",
            6]

        with mock.patch('pycake.helpers.psp_fit.optimize.leastsq', return_value=fake_ret):
            success, fitres, cov, red_chi2 = fit(
                DoubleExponentialPSP(),
                times,
                voltage,
                noise,
                fail_on_negative_cov=[True, True, True, False, False])

            self.assertEqual(success, False)


        # good result and valid ier -> success must equal True
        fake_ret = [
            numpy.array([
                height,
                tau_1,
                tau_2,
                start,
                offset,]),
            numpy.ones((5, 5)),
            "",
            "",
            2]

        with mock.patch('pycake.helpers.psp_fit.optimize.leastsq', return_value=fake_ret):
            success, fitres, cov, red_chi2 = fit(
                DoubleExponentialPSP(),
                times,
                voltage,
                noise,
                fail_on_negative_cov=[True, True, True, False, False])

            self.assertEqual(success, True)


        # bad result (large chi2) but valid ier -> success == False
        fake_ret = [
            numpy.array([
                height * 100.,
                tau_1,
                tau_2,
                start,
                offset + 100.,]),
            numpy.ones((5, 5)),
            "",
            "",
            2]

        with mock.patch('pycake.helpers.psp_fit.optimize.leastsq', return_value=fake_ret):
            success, fitres, cov, red_chi2 = fit(
                DoubleExponentialPSP(),
                times,
                voltage,
                noise,
                fail_on_negative_cov=[True, True, True, False, False])

            self.assertEqual(success, False)

        # good result, negative covariance matrix -> success == False
        fake_ret = [
            numpy.array([
                height,
                tau_1,
                tau_2,
                start,
                offset,]),
            -1.0 * numpy.ones((5, 5)),
            "",
            "",
            2]

        with mock.patch('pycake.helpers.psp_fit.optimize.leastsq', return_value=fake_ret):
            success, fitres, cov, red_chi2 = fit(
                DoubleExponentialPSP(),
                times,
                voltage,
                noise,
                fail_on_negative_cov=[True, True, True, False, False])

            self.assertEqual(success, False)

        # no covariance matrix -> success == False
        fake_ret = [
            numpy.array([
                height,
                tau_1,
                tau_2,
                start,
                offset,]),
            None,
            "",
            "",
            2]

        with mock.patch('pycake.helpers.psp_fit.optimize.leastsq', return_value=fake_ret):
            success, fitres, cov, red_chi2 = fit(
                DoubleExponentialPSP(),
                times,
                voltage,
                noise,
                fail_on_negative_cov=[True, True, True, False, False])

            self.assertEqual(success, False)


if __name__ == '__main__':
    import pylogging
    pylogging.set_loglevel(psp_fit.logger, pylogging.LogLevel.INFO)
    pylogging.append_to_cout(psp_fit.logger)
    unittest.main()
