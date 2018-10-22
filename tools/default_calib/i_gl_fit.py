#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from scipy.optimize import curve_fit

from pycalibtic import NegativePowersPolynomial

parser = argparse.ArgumentParser("Fit tau_m vs DAC")
parser.add_argument('file', type=argparse.FileType('r'),
                    help="file containing tau_m DAC per line")
parser.add_argument('--show_plot', default=False,
                    action="store_true", help="show plot")
parser.add_argument('--plotfilename', default=None,
                    help="store plot to given file")
parser.add_argument('--paramfilename', default=None,
                    help="store fit parameters to given file,"
                    "printed to stdout if not given")
args = parser.parse_args()

data = np.loadtxt(args.file.name)
tau_ms = data[:, 0]
tau_ms_std = data[:, 1]
DACs = data[:, 2]



def tau_m_to_dac(tau_m, param1, param2, param3, param4):
    """
    Cf. calibtic::trafo::NegativePowersPolynomial

    p0 heuristically forced to 0 to improve the fit especially
    in the long tau_m tail.
    """

    parameters = [0, param1, param2, param3, param4]

    oop = NegativePowersPolynomial(parameters, 1e-10, 1)

    try:
        return map(oop.apply, tau_m)
    except TypeError:
        return oop.apply(tau_m)

popt, pcov = curve_fit(tau_m_to_dac, tau_ms, DACs)

if args.paramfilename:
    with open(args.paramfilename, 'w') as f:
        print(",\n".join(map(str, [0] + list(popt))), file=f)
else:
    print("Fitted parameters: {}".format(list(popt)))

if args.show_plot or args.plotfilename:
    import matplotlib.pyplot as plt
    tau_ms_for_fit_plot = np.arange(0.5e-6, max(tau_ms), 1e-8)
    plt.errorbar(tau_ms*1e6, DACs, xerr=tau_ms_std*1e6, label=args.file.name, marker='o', ms=3, errorevery=10)
    plt.plot(tau_ms_for_fit_plot*1e6, tau_m_to_dac(tau_ms_for_fit_plot, *popt),
             label="fit to {}".format(args.file.name), lw=5)
    plt.xlabel("tau_m [us]")
    plt.ylabel("DAC")
    plt.ylim(0,1024)
    plt.xlim(0,max(tau_ms)*1e6)
    plt.legend()
    plt.tight_layout()
    if args.plotfilename:
        plt.savefig(args.plotfilename)
    if args.show_plot:
        plt.show()
