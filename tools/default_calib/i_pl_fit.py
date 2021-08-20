#!/usr/bin/env python

"""
Two separate ranges (see heuristic limits below) are fitted
for a closer match to the data.
"""




import sys

import argparse
import numpy as np
from scipy.optimize import curve_fit, fminbound

from helpers import one_over_polyomial
from pycalibtic import NegativePowersPolynomial

parser = argparse.ArgumentParser("Fit tau_refrac vs DAC")
parser.add_argument('file', type=argparse.FileType('r'),
                    help="file containing tau_refrac DAC per line")
parser.add_argument('--show_plot', default=False,
                    action="store_true", help="show plot")
parser.add_argument('--plotfilename', default=None,
                    help="store plot to given file")
parser.add_argument('--outputfilename', default=None,
                    help="store lookup table to given file"
                    "printed to stdout if not given")
parser.add_argument('--domainfilename', default=None,
                    help="store domain to given file"
                    "printed to stdout if not given")

args = parser.parse_args()


def tau_refrac_to_dac_first_part(tau_ref, param1, param2, param3, param4):
    """
    p0 heuristically forced to 0 to improve the fit especially
    in the long tau_refrac tail.
    """

    parameters = [0, param1, param2, param3, param4]

    oop = NegativePowersPolynomial(parameters, 1e-10, 1)

    try:
        return list(map(oop.apply, tau_ref))
    except TypeError:
        return oop.apply(tau_ref)

def tau_refrac_to_dac_second_part(tau_ref,
                                  param0, param1, param2, param3, param4, param5):

    parameters = [param0, param1, param2, param3, param4, param5]

    oop = NegativePowersPolynomial(parameters, 1e-10, 1)

    try:
        return list(map(oop.apply, tau_ref))

    except TypeError:
        return oop.apply(tau_ref)

# heuristic limits
tau_refrac_limit_first_part = (0.055e-6, 2e-6)
tau_refrac_limit_second_part = (0.005e-6, 0.06e-6)

# load data and separate ranges
data = np.loadtxt(args.file.name)
tau_refracs = data[:, 0]
tau_refracs_std = data[:, 1]
DACs = data[:, 2]

data_first_part = data[np.where(
    np.logical_and(tau_refrac_limit_first_part[0] < tau_refracs,
                   tau_refracs < tau_refrac_limit_first_part[1]))]

tau_refracs_first_part = data_first_part[:, 0]
DACs_first_part = data_first_part[:, 2]

data_second_part = data[np.where(
    np.logical_and(tau_refrac_limit_second_part[0] < tau_refracs,
                   tau_refracs < tau_refrac_limit_second_part[1]))]

tau_refracs_second_part = data_second_part[:, 0]
DACs_second_part = data_second_part[:, 2]

# fit both ranges
popt_first_part, pcov_first_part = curve_fit(
    tau_refrac_to_dac_first_part, tau_refracs_first_part, DACs_first_part)
popt_second_part, pcov_first_part = curve_fit(
    tau_refrac_to_dac_second_part, tau_refracs_second_part, DACs_second_part)

# construct lookup table from DAC to tau_refrac
tau_refrac_join = 0.056e-6
upper_DAC_limit = int(tau_refrac_to_dac_first_part(
    tau_refrac_join, *popt_first_part))

joined_DACs = list(range(11, 1024))
joined_tau_refracs = []

for DAC in range(min(joined_DACs), upper_DAC_limit):
    # heuristic brackets and tolerance
    minimum = fminbound(lambda tau_refrac: (tau_refrac_to_dac_first_part(
        tau_refrac, *popt_first_part) - DAC)**2, tau_refrac_join, 2e-6, xtol=1e-20)
    joined_tau_refracs.append(minimum)

for DAC in range(upper_DAC_limit, max(joined_DACs)+1):
    # heuristic brackets and tolerance
    minimum = fminbound(lambda tau_refrac: (tau_refrac_to_dac_second_part(
        tau_refrac, *popt_second_part) - DAC)**2,
                        0.01e-6, tau_refrac_join, xtol=1e-20)
    joined_tau_refracs.append(minimum)

if not np.all(np.diff(joined_tau_refracs) <= 0):
    raise RuntimeError("Joined tau_refracs not strictly decreasing. Check fit!")

# write output
f = open(args.outputfilename, 'w') if args.outputfilename else sys.stdout
for DAC, tau_refrac in zip(joined_DACs, joined_tau_refracs):
    print("{},".format(tau_refrac), file=f)

# plot if requested
if args.show_plot or args.plotfilename:
    import matplotlib.pyplot as plt
    plt.errorbar(tau_refracs*1e6, DACs, xerr=tau_refracs_std*1e6,
                 label=args.file.name, marker='o', ms=3, errorevery=10)

    plt.plot(tau_refracs_first_part*1e6,
             tau_refrac_to_dac_first_part(tau_refracs_first_part,
                                          *popt_first_part),
             label="fit to {}".format(args.file.name), lw=4)

    plt.plot(tau_refracs_second_part*1e6,
             tau_refrac_to_dac_second_part(tau_refracs_second_part,
                                           *popt_second_part),
             label="fit to {}".format(args.file.name), lw=4)

    plt.plot(np.array(joined_tau_refracs)*1e6, joined_DACs, marker='o', ms=3,
             label="joined lookup from fits")

    plt.xlabel("tau_refrac [us]")
    plt.ylabel("DAC")
    plt.legend()
    plt.xlim(0.001, 2)
    plt.tight_layout()
    if args.plotfilename:
        plt.savefig(args.plotfilename)
    if args.show_plot:
        plt.show()
