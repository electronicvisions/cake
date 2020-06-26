#!/usr/bin/env python

"""
Test the calibtic helper fitting a PSP by integrating the COBA ODE
"""

import unittest
import numpy as np
from pycake.helpers import ode_psp_fit


class TestCalibticHelper(unittest.TestCase):
    def setUp(self):
        self.data = np.loadtxt("share/cake/test_data/ode_psp_fit_membrane.dat")

    def test_fit(self):
        t = np.arange(0, 60, 0.1)
        v = self.data[:, 1]

        orig_v_rest = -50
        orig_e_rev = 20
        orig_tau_rev = 2
        orig_tau_m = 5
        orig_cm = 0.2
        orig_w = 0.01

        # we pin the reversal potential to 20 (mV) as weight and
        # reversal potential are highly correlated
        bounds = [(-np.inf, orig_e_rev - 0.01, 0, 0, 0),
                  (np.inf, orig_e_rev + 0.01, np.inf, np.inf, np.inf)]

        # some very rough starting values
        p0 = [0, 20, 1, 1, 1]

        popt, pcov = ode_psp_fit.fit(t, v, p0, bounds=bounds)

        v_rest, e_rev, tau_rev, tau_m, w_per_cm = popt

        self.assertAlmostEqual(v_rest, orig_v_rest, 2)
        self.assertAlmostEqual(e_rev, orig_e_rev, 1)
        self.assertLess(np.abs(tau_rev - orig_tau_rev) / orig_tau_rev, 0.1)
        self.assertLess(np.abs(tau_m - orig_tau_m) / orig_tau_m, 0.1)
        self.assertAlmostEqual(w_per_cm, orig_w / orig_cm, 2)


if __name__ == "__main__":
    unittest.main()
