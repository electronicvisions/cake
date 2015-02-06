#!/usr/bin/env python
#
# Summarize the calibration in plots
#
# see:
# Heterogeneity and calibration of analog neuromorphic circuits
# https://brainscales.kip.uni-heidelberg.de/internal/jss/AttendMeeting?m=displayPresentation&mI=53&mEID=1738
#
# THIS SCRIPT NEEDS A MAJOR CLEANUP!
#
# everything below exit(0) is not yet generalized:
# - trial-to-trial
# - tau_refrac
# - tau_m

import matplotlib

# http://stackoverflow.com/a/4935945/1350789
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import pycake
from pycake.reader import Reader
from pycake.helpers.peakdetect import peakdet
import pycake.helpers.calibtic as calibtic
import Coordinate as C
import pyhalbe
from pycake.calibration.E_l_I_gl_fixed import E_l_I_gl_fixed_Calibrator
import numpy as np
import os
from collections import defaultdict
import argparse
from pycake.helpers.misc import mkdir_p

font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 12}
matplotlib.rc('font', **font)
margins={"left":0.11, "right":0.95, "top":0.95, "bottom":0.11}

parser = argparse.ArgumentParser()
parser.add_argument("runner", help="path to calibration runner (can be empty)")
parser.add_argument("testrunner", help="path to test runner (evaluation of calibration) (can be empty)")
parser.add_argument("hicann", help="HICANNOnWafer enum", type=int)
parser.add_argument("backenddir", help="path to backends directory (can be empty)")
parser.add_argument("--wafer", help="Wafer enum", default=0, type=int)
parser.add_argument("--outdir", help="path of output directory for plots", default="./figures")

parser.add_argument("--v_reset_runner", help="path to V reset runner (if different from 'runner')", default=None)
parser.add_argument("--v_reset_testrunner", help="path to V reset test runner (if different from 'testrunner')", default=None)

parser.add_argument("--v_t_runner", help="path to V t runner (if different from 'runner')", default=None)
parser.add_argument("--v_t_testrunner", help="path to V t test runner (if different from 'testrunner')", default=None)

parser.add_argument("--e_synx_runner", help="path to E synx runner (if different from 'runner')", default=None)
parser.add_argument("--e_synx_testrunner", help="path to E synx test runner (if different from 'testrunner')", default=None)

parser.add_argument("--e_syni_runner", help="path to E syni runner (if different from 'runner')", default=None)
parser.add_argument("--e_syni_testrunner", help="path to E syni test runner (if different from 'testrunner')", default=None)

parser.add_argument("--e_l_runner", help="path to E l runner (if different from 'runner')", default=None)
parser.add_argument("--e_l_testrunner", help="path to E l test runner (if different from 'testrunner')", default=None)

parser.add_argument("--v_syntcx_runner", help="path to V syntcx runner (if different from 'runner')", default=None)
parser.add_argument("--v_syntcx_testrunner", help="path to V syntcx test runner (if different from 'testrunner')", default=None)

parser.add_argument("--v_syntci_runner", help="path to V syntci runner (if different from 'runner')", default=None)
parser.add_argument("--v_syntci_testrunner", help="path to V syntci test runner (if different from 'testrunner')", default=None)

parser.add_argument("--spikes_testrunner", help="path to spikes test runner (if different from 'testrunner')", default=None)

parser.add_argument("--neuron_enum", help="neuron used for plots", default=0, type=int)

args = parser.parse_args()

fig_dir = args.outdir
mkdir_p(fig_dir)

try:
    reader = Reader(args.runner)
except Exception as e:
    print "Cannot instantiate reader {} because: {}".format(args.runner, e)
    print "Ok, if other runners are specified."
    reader = None

try:
    test_reader = Reader(args.testrunner)
except Exception as e:
    print "Cannot instantiate test reader {} because: {}".format(args.testrunner, e)
    print "Ok, if other runners are specified."
    test_reader = None

def uncalibrated_hist(xlabel, reader, **reader_kwargs):

    if not reader: return

    print "uncalibrated hist for", reader_kwargs["parameter"]

    reader_kwargs["alpha"] = 0.8

    for include_defects in [True, False]:

        reader.include_defects = include_defects

        fig, hists = reader.plot_hists(**reader_kwargs)
        plt.title("uncalibrated", x=0.125, y=0.9)
        plt.xlabel(xlabel)
        plt.ylabel("#")
        plt.xlim(*reader_kwargs['range'])
        plt.subplots_adjust(**margins)

        defects_string = "with_defects" if include_defects else "without_defects"

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "uncalibrated.pdf"])))

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "uncalibrated.png"])))

        #--------------------------------------------------------------------------------

        fig, foos = reader.plot_vs_neuron_number_s(**reader_kwargs)
        plt.title("uncalibrated", x=0.125, y=0.9)
        plt.xlabel("Neuron")
        plt.ylabel(xlabel)
        plt.xlim(0, 512)
        plt.subplots_adjust(**margins)

        defects_string = "with_defects" if include_defects else "without_defects"

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "uncalibrated_vs_neuron_number.pdf"])))

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "uncalibrated_vs_neuron_number.png"])))

        #--------------------------------------------------------------------------------

        fig, foos = reader.plot_vs_neuron_number_s(sort_by_shared_FG_block=True, **reader_kwargs)
        plt.title("uncalibrated", x=0.125, y=0.9)
        plt.xlabel("Shared FG block*128 + Neuron%256/2")
        plt.ylabel(xlabel)
        plt.xlim(0, 512)
        plt.subplots_adjust(**margins)

        defects_string = "with_defects" if include_defects else "without_defects"

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "uncalibrated_vs_shared_FG_block.pdf"])))

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "uncalibrated_vs_shared_FG_block.png"])))


def calibrated_hist(xlabel, reader, **reader_kwargs):

    if not reader: return

    print "calibrated hist for", reader_kwargs["parameter"]

    reader_kwargs["alpha"] = 0.8

    for include_defects in [True, False]:

        reader.include_defects = include_defects

        fig, hists = reader.plot_hists(**reader_kwargs)
        plt.title("calibrated", x=0.125, y=0.9)
        plt.xlabel(xlabel)
        plt.ylabel("#")
        plt.xlim(*reader_kwargs['range'])
        plt.subplots_adjust(**margins)

        defects_string = "with_defects" if include_defects else "without_defects"

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "calibrated.pdf"])))

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "calibrated.png"])))

        # --------------------------------------------------------------------------------

        fig, foos = reader.plot_vs_neuron_number_s(**reader_kwargs)
        plt.title("calibrated", x=0.125, y=0.9)
        plt.xlabel("Neuron")
        plt.ylabel(xlabel)
        plt.xlim(0,512)
        plt.subplots_adjust(**margins)

        defects_string = "with_defects" if include_defects else "without_defects"

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "calibrated_vs_neuron_number.pdf"])))

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "calibrated_vs_neuron_number.png"])))

        #--------------------------------------------------------------------------------

        fig, foos = reader.plot_vs_neuron_number_s(sort_by_shared_FG_block=True, **reader_kwargs)
        plt.title("uncalibrated", x=0.125, y=0.9)
        plt.xlabel("Shared FG block*128 + Neuron%256/2")
        plt.ylabel(xlabel)
        plt.xlim(0, 512)
        plt.subplots_adjust(**margins)

        defects_string = "with_defects" if include_defects else "without_defects"

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "calibrated_vs_shared_FG_block.pdf"])))

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "calibrated_vs_shared_FG_block.png"])))


def trace(ylabel, reader, parameter, neuron, steps=None, start=0, end=-1, suffix=""):

    if not reader: return

    print "trace for", parameter

    fig = plt.figure()

    e = reader.runner.experiments[parameter]

    if steps == None:
        steps = range(len(e.measurements))

    for step in steps:

        m = e.measurements[step]
        t = m.get_trace(neuron)

        if not t:
            print "missing trace for", parameter, neuron, step
            return

        plt.plot(np.array(t[0][start:end])*1e6, t[1][start:end]*1000);

    plt.grid(True)

    plt.xlabel("t [$\mu$s]")
    plt.ylabel(ylabel)
    plt.subplots_adjust(**margins)
    plt.savefig(os.path.join(fig_dir,parameter+"_trace"+suffix+".pdf"))
    plt.savefig(os.path.join(fig_dir,parameter+"_trace"+suffix+".png"))

def result(label, xlabel=None, ylabel=None, reader=None, suffix="", ylim=None, **reader_kwargs):
    """ label must have placeholder 'inout' for 'in' and 'out' x and y labels, 
        like: '$E_{{synx}}$ {inout} [mV]'
    """

    if not reader: return

    print "result for", reader_kwargs["parameter"]

    for include_defects in [True, False]:

        reader.include_defects = include_defects

        fig = reader.plot_result(**reader_kwargs)
        plt.xlabel(xlabel if xlabel != None else label.format(inout="(in)"))
        plt.ylabel(ylabel if ylabel != None else label.format(inout="(out)"))

        plt.subplots_adjust(**margins)

        if ylim:
            plt.ylim(*ylim)

        defects_string = "with_defects" if include_defects else "without_defects"

        plt.savefig(os.path.join(fig_dir,"_".join([reader_kwargs["parameter"],
                                                   defects_string,
                                                   "result"+suffix+".pdf"])))

        plt.savefig(os.path.join(fig_dir,"_".join([reader_kwargs["parameter"],
                                                   defects_string,
                                                   "result"+suffix+".png"])))

if args.backenddir:

    # offset

    fig = plt.figure()

    c = calibtic.Calibtic(args.backenddir,C.Wafer(C.Enum(args.wafer)),C.HICANNOnWafer(C.Enum(args.hicann)))

    def get_offset(cal, nrnidx):
        try:
            offset = cal.nc.at(nrnidx).at(21).apply(0)
            return offset
        except IndexError:
            print "No offset found for Neuron {}. Is the wafer and hicann enum correct? (w{}, h{})".format(nrnidx,args.wafer,args.hicann)
            return 0

    offsets = [get_offset(c, n) * 1000 for n in xrange(512)]
    plt.hist(offsets, bins=100);
    plt.xlabel("offset [mV]")
    plt.ylabel("#")
    plt.subplots_adjust(**margins)
    plt.savefig(os.path.join(fig_dir,"analog_readout_offset.pdf"))
    plt.savefig(os.path.join(fig_dir,"analog_readout_offset.png"))

## V reset

r_v_reset = reader if args.v_reset_runner == None else Reader(args.v_reset_runner)

if r_v_reset:

    uncalibrated_hist("$V_{reset}$ [V]",
                      r_v_reset,
                      parameter="V_reset",
                      key="baseline",
                      bins=100,
                      range=(0.4,0.8),
                      show_legend=False)

    trace("$V_{mem}$ [mV]", r_v_reset, "V_reset", C.NeuronOnHICANN(C.Enum(args.neuron_enum)), end=2000, suffix="_uncalibrated")

    result("$V_{{reset}}$ {inout} [mV]", reader=r_v_reset, parameter="V_reset",key="baseline",alpha=0.05)

r_test_v_reset = test_reader if args.v_reset_testrunner == None else Reader(args.v_reset_testrunner)

if r_test_v_reset:

    calibrated_hist("$V_{reset}$ [V]",
                      r_test_v_reset,
                      parameter="V_reset",
                      key="baseline",
                      bins=100,
                      range=(0.46,0.54),
                      show_legend=True)

    trace("$V_{mem}$ [mV]", r_test_v_reset, "V_reset", C.NeuronOnHICANN(C.Enum(args.neuron_enum)), end=510, suffix="_calibrated")

## E synx

r_e_synx = reader if args.e_synx_runner == None else Reader(args.e_synx_runner)

if r_e_synx:

    uncalibrated_hist("$E_{synx}$ [V]",
                      r_e_synx,
                      parameter="E_synx",
                      key="mean",
                      bins=100,
                      range=(0.55,0.99),
                      show_legend=False);

    result("$E_{{synx}}$ {inout} [mV]", reader=r_e_synx, ylim=[500,1000], parameter="E_synx",key="mean",alpha=0.05)

    trace("$V_{mem}$ [mV]", r_e_synx, "E_synx", C.NeuronOnHICANN(C.Enum(args.neuron_enum)), end=510, suffix="_uncalibrated")

r_test_e_synx = test_reader if args.e_synx_testrunner == None else Reader(args.e_synx_testrunner)

if r_test_e_synx:

    calibrated_hist("$E_{synx}$ [V]",
                    r_test_e_synx,
                    parameter="E_synx",
                    key="mean",
                    bins=100,
                    range=(0.55,0.95),
                    show_legend=True);

    result("$E_{{synx}}$ {inout} [mV]", reader=r_test_e_synx, suffix="_calibrated", ylim=[500,1000], parameter="E_synx",key="mean",alpha=0.05)

    trace("$V_{mem}$ [mV]", r_test_e_synx, "E_synx", C.NeuronOnHICANN(C.Enum(args.neuron_enum)), end=510, suffix="_calibrated")

#for k,v in r_test_e_synx.get_results("E_synx", r_test_e_synx.get_neurons(), "mean").iteritems():
#        if v[0] < 0.78 or v[0] > 0.82:
#            print k, k.id(), v[0]

## E syni

r_e_syni = reader if args.e_syni_runner == None else Reader(args.e_syni_runner)

if r_e_syni:

    uncalibrated_hist("$E_{syni}$ [V]",
                      r_e_syni,
                      parameter="E_syni",
                      key="mean",
                      bins=100,
                      range=(0.4, 0.8),
                      show_legend=False);

    result("$E_{{syni}}$ {inout} [mV]", reader=r_e_syni, ylim=[400,900], parameter="E_syni",key="mean",alpha=0.05)

    trace("$V_{mem}$ [mV]", r_e_syni, "E_syni", C.NeuronOnHICANN(C.Enum(args.neuron_enum)), end=510, suffix="_uncalibrated")

r_test_e_syni = test_reader if args.e_syni_testrunner == None else Reader(args.e_syni_testrunner)

if r_test_e_syni:

    calibrated_hist("$E_{syni}$ [V]",
                    r_test_e_syni,
                    parameter="E_syni",
                    key="mean",
                    bins=100,
                    range=(0.55, 0.95),
                    show_legend=True);

    result("$E_{{syni}}$ {inout} [mV]", reader=r_test_e_syni, suffix="_calibrated", ylim=[400,900], parameter="E_syni",key="mean",alpha=0.05)

    trace("$V_{mem}$ [mV]", r_test_e_syni, "E_syni", C.NeuronOnHICANN(C.Enum(args.neuron_enum)), end=510, suffix="_calibrated")

## E l

r_e_l = reader if args.e_l_runner == None else Reader(args.e_l_runner)

if r_e_l:

    uncalibrated_hist("$E_{l}$ [V]",
                      r_e_l,
                      parameter="E_l",
                      key="mean",
                      bins=100,
                      range=(0.5,1),
                      show_legend=False)

    result("$E_{{l}}$ {inout} [mV]", reader=r_e_l, ylim=[400,900], parameter="E_l",key="mean",alpha=0.05)

    trace("$V_{mem}$ [mV]", r_e_l, "E_l", C.NeuronOnHICANN(C.Enum(args.neuron_enum)), end=510, suffix="_uncalibrated")

    """

    r_e_l.include_defects = True
    r_e_l.plot_result("E_l","mean");

    r_e_l.include_defects = False

    neurons = r_e_l.get_neurons()[132:135]

    fig = r_e_l.plot_result("E_l","mean",neurons,marker='o',linestyle="None");

    print r_e_l.runner.coeffs.keys()

    coeffs = r_e_l.runner.coeffs["E_l"]

    xs = np.array([500,900])

    for n in neurons:
        #print len(coeffs), len(coeffs[0]), len(coeffs[0][1])
        c = coeffs[0][1][n]
        if c == None: 
            #print c
            #continue
            pass
        a = c[0]
        b = c[1]
        #print a,b, 1/a, -b/a
        polynomial = numpy.poly1d([1/a,-b/a])
        #print polynomial
        plt.plot(xs,np.array(polynomial(xs/1800.*1023.)*1800./1023.), label="Neuron {}".format(n.id().value()))

    plt.xlabel("Input [DAC]")
    plt.ylabel("Output [mV]")
    plt.subplots_adjust(**margins)
    plt.xlim(500,900)
    plt.ylim(500,900)
    plt.legend(loc="upper left")
    plt.grid(True)
    fig.savefig(os.path.join(fig_dir,"calib_example_lines.pdf"))
    """

r_test_e_l = test_reader if args.e_l_testrunner == None else Reader(args.e_l_testrunner)

if r_test_e_l:

    calibrated_hist("$E_{l}$ [V]",
                    r_test_e_l,
                    parameter="E_l",
                    key="mean",
                    show_legend=True,
                    bins=100,
                    range=(0.6,0.8))

    result("$E_{{l}}$ {inout} [mV]", reader=r_test_e_l, suffix="_calibrated", ylim=[400,900], parameter="E_l",key="mean",alpha=0.05)

    trace("$V_{mem}$ [mV]", r_test_e_l, "E_l", C.NeuronOnHICANN(C.Enum(args.neuron_enum)), end=510, suffix="_calibrated")

## V t

r_v_t = reader if args.v_t_runner == None else Reader(args.v_t_runner)

if r_v_t:

    uncalibrated_hist("$V_{t}$ [V]",
                      r_v_t,
                      parameter="V_t",
                      key="max",
                      bins=100,
                      range=(0.5,0.85))

    result("$V_{{t}}$ {inout} [mV]", reader=r_v_t, ylim=[500,1000], parameter="V_t",key="max",alpha=0.05)

    trace("$V_{mem}$ [mV]", r_v_t, "V_t", C.NeuronOnHICANN(C.Enum(args.neuron_enum)), end=510, suffix="_uncalibrated")

r_test_v_t = test_reader if args.v_t_testrunner == None else Reader(args.v_t_testrunner)

if  r_test_v_t:

    calibrated_hist("$V_{t}$ [V]",
                    r_test_v_t,
                    parameter="V_t",
                    key="max",
                    bins=100,
                    range=(0.65,0.85),
                    show_legend=True)

    #for k,v in r_test_v_t.get_results("V_t", range(512), "max").iteritems():
    #        if v[0] < 0.675 or v[0] > 0.725:
    #            print 0.7, k.id(), v[0]
    #            print
    #        if v[1] < 0.725 or v[1] > 0.775:
    #            print 0.75, k.id(), v[1]
    #            print
    #        if v[2] < 0.775 or v[2] > 0.825:
    #            print 0.8, k.id(), v[2]
    #            print

    result("$V_{{t}}$ {inout} [mV]", reader=r_test_v_t, suffix="_calibrated", ylim=[500,1000], parameter="V_t",key="max",alpha=0.05)

    trace("$V_{mem}$ [mV]", r_test_v_t, parameter="V_t", neuron=C.NeuronOnHICANN(C.Enum(args.neuron_enum)), start=500, end=700, suffix="_calibrated")

    #r_v_t.include_defects = False

    #neurons = r_v_t.get_neurons()[0:1]

    #fig = r_v_t.plot_result("V_t","max",neurons,marker='o',linestyle="None");

    #print r_v_t.runner.coeffs.keys()

    #coeffs = r_v_t.runner.coeffs["V_t"]
    #
    #xs = np.array([550,850])
    #
    #for n in neurons:
    #    #print len(coeffs), len(coeffs[0]), len(coeffs[0][1])
    #    c = coeffs[0][1][n]
    #    if c == None: 
    #        #print c
    #        continue
    #        pass
    #    a = c[0]
    #    b = c[1]
    #    #print a,b, 1/a, -b/a
    #    polynomial = numpy.poly1d([1/a,-b/a])
    #    #print polynomial
    #    plt.plot(xs,np.array(polynomial(xs/1800.*1023.)*1800./1023.))

## E l, I gl

# In[72]:

#r_e_l_i_gl = #Reader("/afsuser/sschmitt/.calibration-restructure/runner_E_l_I_gl_fixed_0527_1211.p.bz2")


# In[76]:

#e=r_e_l_i_gl.runner.experiments["E_l_I_gl_fixed"]
#m=e.measurements[-1]
#calibrator = E_l_I_gl_fixed_Calibrator(e)
#print calibrator.fit_neuron(C.NeuronOnHICANN(C.Enum(1)))


## V syntcx psp max

r_v_syntcx = reader if args.v_syntcx_runner == None else Reader(args.v_syntcx_runner)

if r_v_syntcx:

    """

    e = r_v_syntcx.runner.experiments["V_syntcx_psp_max"]

    x = 3000

    neuron = C.NeuronOnHICANN(C.Enum(103))

    for m in [e.measurements[idx] for idx in [0,-1,3]]:
        t = m.get_trace(neuron)
        plt.plot(np.array(t[0][:x])*1e6,t[1][:x], label="$V_{{syntcx}}$ {:.0f} [mV]".format(#np.std(t[1]), 
                 m.sthal.hicann.floating_gates.getNeuron(neuron, pyhalbe.HICANN.neuron_parameter.V_syntcx)/1023.*1800))
    plt.legend()
    plt.ylabel("$V_{mem}$ [mV]")
    plt.xlabel("t [$\mu$s]")
    plt.ylim(0.69, 0.78)
    plt.subplots_adjust(**margins)
    plt.savefig(os.path.join(fig_dir,"V_syntcx_trace.pdf"))

    data = t[1]

    max_std = -1
    current_std = 0

    period_index = 6000

    stds = []
    period_indices = []

    for period_index in range(1,len(data)):

        nperiods = len(data)/period_index
        max_index = nperiods*period_index
        data_cut = data[:max_index]
        data_split = np.array_split(data_cut, len(data_cut)/period_index)
        avg_raw = np.mean([ds for ds in data_split], axis=0)
        current_std = np.std(avg_raw)

        period_indices.append(period_index)
        stds.append(current_std)

    plt.plot(period_indices, stds)

    #plt.scatter(np.array(maxtab)[:,0][:x], np.array(maxtab)[:,1][:x], color='blue')


    maxtab, mintab = peakdet(stds, 0.003)

    # In[57]:

    data = t[1]

    period_index = 959

    nperiods = len(data)/period_index
    max_index = nperiods*period_index
    data_cut = data[:max_index]
    data_split = np.array_split(data_cut, len(data_cut)/period_index)
    avg_raw = np.mean([ds for ds in data_split], axis=0)
    current_std = np.std(avg_raw)

    plt.plot(avg_raw)

    print np.std(avg_raw), np.std(t[1])

    maxtab, mintab = peakdet(avg_raw, np.std(avg_raw))

    print maxtab, mintab

    plt.scatter(np.array(maxtab)[:,0], np.array(maxtab)[:,1], color='red')

    def my_exp(x):
        return 0.745 + 0.04*(np.exp(-1/70.*(x-113)) - 1)

    exp_range = np.arange(113,400)

    plt.plot(exp_range, my_exp(exp_range))

    # In[43]:

    m.sthal.hicann.floating_gates.getNeuron(neuron, pyhalbe.HICANN.neuron_parameter.V_syntcx)

    """

    # In[222]:
    #r_v_syntcx.plot_hists("V_syntcx_psp_max", "std", bins=100, range=(0,0.03), draw_target_line=False);

    # In[223]:
    #r_v_syntcx.plot_result("V_syntcx_psp_max","mean", color='b', alpha=0.1);

    # In[310]:

    result(label=None,
           xlabel="$V_{syntcx}$ [mV]",
           ylabel="$\sigma$(trace) [mV]",
           reader=r_v_syntcx,
           parameter="V_syntcx_psp_max",
           key="std",
           neurons=range(50),
           mark_top_bottom=False,
           alpha=0.5)


    trace("$V_{mem}$ [mV]", r_v_syntcx, "V_syntcx_psp_max", C.NeuronOnHICANN(C.Enum(args.neuron_enum)), end=4000)

    # In[311]:

    fig = plt.figure()
    r_v_syntcx.include_defects = False
    results_v_syntcx = r_v_syntcx.get_results("V_syntcx_psp_max",r_v_syntcx.get_neurons(),"std")
    max_stds = [np.max(stds)*1000 for n, stds in results_v_syntcx.iteritems()]
    plt.hist(max_stds,bins=100);
    plt.xlabel("$\sigma$(trace) [mV]")
    plt.ylabel("#")
    plt.ylim(0,23)
    plt.subplots_adjust(**margins)
    plt.savefig(os.path.join(fig_dir,"V_syntcx_psp_stds.pdf"))
    plt.savefig(os.path.join(fig_dir,"V_syntcx_psp_stds.png"))

    # In[242]:

    sum(np.array(max_stds) < 0.005)

## V syntci psp max

r_v_syntci = reader if args.v_syntci_runner == None else Reader(args.v_syntci_runner)

if r_v_syntci:

    e = r_v_syntci.runner.experiments["V_syntci_psp_max"]

    """

    x = 3000

    neuron = C.NeuronOnHICANN(C.Enum(401))

    for m in [e.measurements[idx] for idx in [0,3,-1]]:
        t = m.get_trace(neuron)
        plt.plot(np.array(t[0][:x])*1e6,t[1][:x], label="$V_{{syntci}}$ {:.0f} [mV]".format(#np.std(t[1]), 
                 m.sthal.hicann.floating_gates.getNeuron(neuron, pyhalbe.HICANN.neuron_parameter.V_syntci)/1023.*1800))
    plt.legend()
    plt.ylabel("$V_{mem}$ [mV]")
    plt.xlabel("t [$\mu$s]")
    plt.ylim(0.64, 0.76)
    plt.subplots_adjust(**margins)
    plt.savefig(os.path.join(fig_dir,"V_syntci_trace.pdf"))

    """

    # In[318]:

    result(label=None,
           xlabel="$V_{syntci}$ [mV]",
           ylabel="$\sigma$(trace) [mV]",
           reader=r_v_syntci,
           parameter="V_syntci_psp_max",
           key="std",
           neurons=range(50),
           mark_top_bottom=False,
           alpha=0.5)

    trace("$V_{mem}$ [mV]", r_v_syntci, "V_syntci_psp_max", C.NeuronOnHICANN(C.Enum(args.neuron_enum)), end=510)

    # In[319]:

    fig = plt.figure()
    r_v_syntci.include_defects = False
    results_v_syntci = r_v_syntci.get_results("V_syntci_psp_max",r_v_syntci.get_neurons(),"std")
    max_stds = [np.max(stds)*1000 for n, stds in results_v_syntci.iteritems()]
    plt.hist(max_stds,bins=100);
    plt.xlabel("$\sigma$(trace) [mV]")
    plt.ylabel("#")
    plt.ylim(0,23)
    plt.xlim(0,30)
    plt.subplots_adjust(**margins)
    plt.savefig(os.path.join(fig_dir,"V_syntci_psp_stds.pdf"))
    plt.savefig(os.path.join(fig_dir,"V_syntci_psp_stds.png"))

"""
    # In[247]:

    sum(np.array(max_stds) < 0.005)


    # In[253]:

    bad_syntci = [n  for n, stds in results_v_syntci.iteritems() if np.max(stds) < 0.005];
    bad_syntcx = [n  for n, stds in results_v_syntcx.iteritems() if np.max(stds) < 0.005]


    # In[255]:

    set(bad_syntcx).intersection(bad_syntci)
"""

r_test_spikes = test_reader if args.spikes_testrunner == None else Reader(args.spikes_testrunner)

if r_test_spikes:

    for include_defects in [True, False]:

        defects_string = "with_defects" if include_defects else "without_defects"

        r_test_spikes.include_defects = include_defects

        fig = plt.figure()

        n_spikes = np.array(r_test_spikes.get_results("Spikes",r_test_spikes.get_neurons(), "spikes_n_spikes").values())

        plt.plot([v[pyhalbe.HICANN.neuron_parameter.V_t].value for v in r_test_spikes.runner.config.copy("Spikes").get_steps()], np.sum(np.greater(n_spikes,1), axis=0), label="measured")
        #plt.plot(V_ts, np.array(n_spikes_est), label="estimated")
        plt.axhline(len(r_test_spikes.get_neurons()), color='black', linestyle="dotted")
        plt.legend(loc="lower left")
        plt.ylabel("# spiking neurons")
        plt.xlabel("$V_t$ [mV]")

        plt.subplots_adjust(**margins)
        plt.savefig(os.path.join(fig_dir,"n_spiking_neurons_"+defects_string+"_calibrated.png"))

        # number of spikes

        r_test_spikes.include_defects = include_defects

        fig = r_test_spikes.plot_result("Spikes","spikes_n_spikes",yfactor=1,average=True,mark_top_bottom=True)

        plt.ylabel("average number of recorded spikes per neuron")
        plt.xlabel("$V_t$ [mV]")

        plt.subplots_adjust(**margins)

        defects_string = "with_defects" if include_defects else "without_defects"

        plt.savefig(os.path.join(fig_dir,"_".join(["n_spikes",
                                                   defects_string,
                                                   "result_calibrated.pdf"])))

        plt.savefig(os.path.join(fig_dir,"_".join(["n_spikes",
                                                   defects_string,
                                                   "result_calibrated.png"])))


exit(0)

## Trial to Trial

### E l

# In[247]:

#ttt_e_l_reader = Reader("/afsuser/sschmitt/.calibration-restructure/testrunner_0616_0904.p.bz2") # 10
#ttt_e_l_reader = Reader("/afsuser/sschmitt/.calibration-restructure/testrunner_0616_1206.p.bz2") # 50
ttt_e_l_reader = Reader("/afsuser/sschmitt/.calibration-restructure/testrunner_0616_1352.p.bz2") # 10, smallest I_gl removed
ttt_e_l_reader.include_defects = False


# In[248]:

ttt_e_l_results = ttt_e_l_reader.get_results("E_l", neurons=ttt_e_l_reader.get_neurons(), key="mean")
neurons_means_stds_e_l = [(key, np.mean(values), np.std(values), values) for key, values in ttt_e_l_results.iteritems()]


# In[249]:

plt.hist([np.mean(values) for values in ttt_e_l_results.values()], bins=100);


# In[250]:

fg = ttt_e_l_reader.runner.experiments['E_l'].measurements[0].sthal.hicann.floating_gates
fg.getNeuron(C.NeuronOnHICANN(C.Enum(0)), pyhalbe.HICANN.neuron_parameter.I_gl)


# In[320]:

plt.plot(np.array(range(len(neurons_means_stds_e_l)))/float(len(neurons_means_stds_e_l)), 
         [fg.getNeuron(l[0], 
                    pyhalbe.HICANN.neuron_parameter.I_gl) for l in sorted(neurons_means_stds_e_l, key=lambda l: l[2])])
plt.grid(True)
plt.locator_params(nbins=10)
plt.ylabel("$I_{gl}$ [DAC]")
plt.xlabel("Cummulative on $\sigma(V_{rest})$")
plt.subplots_adjust(**margins)
plt.savefig(os.path.join(fig_dir,"V_rest_ttt_cummulative_I_gl.pdf"))


# In[321]:

plt.plot(np.array(range(len(neurons_means_stds_e_l)))/float(len(neurons_means_stds_e_l)),
         [l[2]*1000 for l in sorted(neurons_means_stds_e_l, key=lambda l: l[2])]
         )
plt.grid(True)
plt.locator_params(nbins=10)
plt.xlabel("Cummulative")
plt.ylabel("$\sigma(V_{rest})$ [mV]")
plt.subplots_adjust(**margins)
plt.savefig(os.path.join(fig_dir,"V_rest_ttt_cummulative.pdf"))
plt.savefig(os.path.join(fig_dir,"V_rest_ttt_cummulative.png"))


# In[12]:

plt.plot(np.array(range(len(neurons_means_stds_e_l)))/float(len(neurons_means_stds_e_l)), 
         [l[0].id().value() for l in sorted(neurons_means_stds_e_l, key=lambda l: l[2])])
plt.grid(True)
plt.locator_params(nbins=10)


# In[9]:

plt.plot(np.array(range(len(neurons_means_stds_e_l)))/float(len(neurons_means_stds_e_l)),
         [l[3] for l in sorted(neurons_means_stds_e_l, key=lambda l: l[2])], marker='None'
         )
plt.grid(True)
plt.locator_params(nbins=10)
plt.xlabel("Cummulative")
plt.ylabel("$\sigma(V_{rest})$ [mV]")
plt.subplots_adjust(**margins)


# In[322]:

bins = numpy.linspace(0.65, 0.75, 100)
for n in [0,200,-1]:
    plt.hist(sorted(neurons_means_stds_e_l, key=lambda l: l[2])[n][3], bins=bins, alpha=0.8);
plt.subplots_adjust(**margins)
plt.savefig(os.path.join(fig_dir,"V_rest_ttt_hist.pdf"))


### E synx

# In[499]:

ttt_e_synx_reader = Reader("/afsuser/sschmitt/.calibration-restructure/testrunner_0616_1137.p.bz2")
ttt_e_synx_reader.include_defects = False
ttt_e_synx_results = ttt_e_synx_reader.get_results("E_synx", neurons=ttt_e_synx_reader.get_neurons(), key="mean")
neurons_means_stds_e_synx = [(key, np.mean(values), np.std(values), values) for key, values in ttt_e_synx_results.iteritems()]


# In[500]:

plt.hist([l[2]*1000 for l in sorted(neurons_means_stds_e_synx, key=lambda l: l[2])], bins=100);


# In[501]:

plt.plot(np.array(range(len(neurons_means_stds_e_synx)))/float(len(neurons_means_stds_e_synx)),
         [l[2]*1000 for l in sorted(neurons_means_stds_e_synx, key=lambda l: l[2])]
         )
plt.grid(True)
plt.locator_params(nbins=10)
plt.xlabel("Cummulative")
plt.ylabel("$\sigma(E_{synx})$ [mV]")
plt.subplots_adjust(**margins)
plt.savefig(os.path.join(fig_dir,"E_synx_ttt_cummulative.pdf"))


# In[ ]:

plt.plot(np.array(range(len(neurons_means_stds)))/float(len(neurons_means_stds)),
         [l[3] for l in sorted(neurons_means_stds, key=lambda l: l[2])], marker='None'
         )


## I gl / tau m

# In[89]:

I_gl_test_reader = test_reader


# In[90]:

m = I_gl_test_reader.runner.experiments["I_gl"].measurements[0]


# In[91]:

t = m.get_trace(C.NeuronOnHICANN(C.Enum(20)))


# In[92]:

x = 5000
plt.plot(t[0][:x], t[1][:x])


# In[144]:

tau_m_results = I_gl_test_reader.get_results("I_gl", neurons=range(512), key="tau_m")


# In[94]:

plt.hist([v[0]*1e6 for v in results.values() if v[0] != None], bins=100);


# In[129]:

I_gls = []
tau_ms = []
tau_ms_I_gl_40 = []
for n in range(512):
    neuron = C.NeuronOnHICANN(C.Enum(n))
    I_gl = m.sthal.hicann.floating_gates.getNeuron(neuron, pyhalbe.HICANN.neuron_parameter.I_gl)
    tau_m = tau_m_results[neuron][0]
    if tau_m != None and I_gl > 0:
        I_gls.append(I_gl)
        tau_ms.append(tau_m*1e6)
        if I_gl == 40:
            tau_ms_I_gl_40.append(tau_m*1e6)


# In[121]:

#print tau_ms
plt.scatter(I_gls, tau_ms)
plt.xlim(0,1000)
plt.ylim(0,100)


# In[351]:

plt.hist(I_gls, bins=100);
plt.xlabel("$I_{gl}$ [DAC]")
plt.ylabel("#")
plt.subplots_adjust(**margins)
plt.savefig(os.path.join(fig_dir,"hist_I_gl.pdf"))


# In[350]:

plt.hist(tau_ms_I_gl_40, bins=np.linspace(0,100,100));
plt.subplots_adjust(**margins)
plt.ylabel("#")
plt.xlabel(r"$\tau_{m}$ [$\mu$s]")
plt.xlim(0,80)
plt.savefig(os.path.join(fig_dir,"tau_m_hist.pdf"))


## tau_syn x

# In[324]:

tau_syn_x_reader = Reader("/afsuser/sschmitt/.calibration-restructure/testrunner_0618_1439.p.bz2")


# In[325]:

e = tau_syn_x_reader.runner.experiments["V_syntcx"]
m = e.measurements[0]


# In[326]:

taus_for_compare = {}

for neuron, result in e.results[0].iteritems():
    fit = result['fit']
    if fit:
        tau_m = tau_m_results[neuron][0]
        taus_for_compare[neuron] = {'tau_1':fit['tau_1']*1e6, 'tau_2':fit['tau_2']*1e6, 'tau_m':tau_m*1e6}


# In[337]:

tau_1_2_closer_to_tau_m = []
tau_1_2_farther_to_tau_m = []

fudge_factor = 0.9

for v in taus_for_compare.values():
    tau_m = fudge_factor*v['tau_m']
    if abs(tau_m - v['tau_1']) < abs(tau_m - v['tau_2']):
        tau_1_2_closer_to_tau_m.append(v['tau_1'])
        tau_1_2_farther_to_tau_m.append(v['tau_2'])
    else:
        tau_1_2_closer_to_tau_m.append(v['tau_2'])
        tau_1_2_farther_to_tau_m.append(v['tau_1'])


plt.scatter(tau_1_2_closer_to_tau_m,
            [fudge_factor*v['tau_m'] for v in taus_for_compare.values()], color='b')

plt.scatter(tau_1_2_farther_to_tau_m,
            [fudge_factor*v['tau_m'] for v in taus_for_compare.values()], color='r')

plt.xlabel("tau_m [$\mu$s")
plt.ylabel("tau_1_2 [$\mu$s]")

pmax = 10

plt.ylim(0,pmax)
plt.xlim(0,pmax)
plt.plot([0,pmax],[0,pmax], color='black')


# In[328]:

len(taus_for_compare)


# In[341]:

for n in range(7,13):
    t = e.results[0][C.NeuronOnHICANN(C.Enum(n))]['trace']
    plt.plot(t[0]*1e6,t[1], label="Neuron {}".format(n))
plt.xlabel("t [$\mu$s]")
plt.ylabel("$V_{mem}$ [V]")
plt.legend()
plt.subplots_adjust(**margins)
plt.savefig(os.path.join(fig_dir,"V_syntcx_psp_max_traces.pdf"))


## I pl / tau refrac

# In[423]:

I_pl_reader = Reader("/afsuser/sschmitt/.calibration-restructure/runner_0624_1419.p.bz2")
e=I_pl_reader.runner.experiments["I_pl"]


# In[424]:

m=e.measurements[0]
m.sthal.hicann.floating_gates.getNeuron(C.NeuronOnHICANN(C.Enum(0)),pyhalbe.HICANN.neuron_parameter.I_pl)


# In[425]:

ref = e.results[0]


# In[471]:

results = defaultdict(list)

for n,meas in enumerate(e.measurements):
    if n == 0: continue
    I_pl = meas.sthal.hicann.floating_gates.getNeuron(C.NeuronOnHICANN(C.Enum(0)),pyhalbe.HICANN.neuron_parameter.I_pl)
    
    print I_pl
    
    for key, value in e.results[n].iteritems():
        ref_tau_refrac = ref[key]['tau_refrac']
        step_tau_refrac = e.results[n][key]['tau_refrac']
    
        if ref_tau_refrac != None and step_tau_refrac != None:
            tau_refrac = (step_tau_refrac-ref_tau_refrac)*1e6
            if tau_refrac > 10:
                print ">15", key, I_pl
            results[I_pl].append(tau_refrac)
        else:
            print key, "fail"


# In[485]:

for I_pl in reversed([20,27,41,82,205]):
    values = results[I_pl]
    key = I_pl
    plt.hist(values, label="$I_{{pl}}$ {:02d} [DAC]".format(key), bins=np.linspace(0,3,100))
plt.legend()
plt.xlabel(r"$\tau_{refrac}$ [$\mu$s]")
plt.ylabel("#")
plt.subplots_adjust(**margins)
plt.savefig(os.path.join(fig_dir,"I_pl_hist.pdf"))


# In[480]:

for I_pl in reversed([4]):
    values = results[I_pl]
    key = I_pl
    plt.hist(values, label="$I_{{pl}}$ {:02d} [DAC]".format(key), bins=np.linspace(0,30,100))
plt.legend()
plt.xlabel(r"$\tau_{refrac}$ [$\mu$s]")
plt.ylabel("#")
#plt.yscale("log")
#plt.yscale("log")


# In[497]:

for m in [e.measurements[_] for _ in [0,-3]]:
    t = m.get_trace(C.NeuronOnHICANN(C.X(215), C.bottom))
    x=1000
    plt.plot(t[0][:x]*1e6,t[1][:x])
plt.ylabel("$V_{mem}$ [mV]")
plt.xlabel("t [$\mu$s]")
plt.subplots_adjust(**margins)
plt.savefig(os.path.join(fig_dir,"I_pl_trace.pdf"))


## misc

# In[364]:

test_e_l_reader_foo = Reader("/afsuser/sschmitt/.calibration-restructure/testrunner_0623_1515.p.bz2")


# In[365]:

test_e_l_results_foo = test_e_l_reader_foo.get_results("E_l", neurons=ttt_e_l_reader.get_neurons(), key="mean")


# In[366]:

plt.hist([np.mean(values) for values in test_e_l_results_foo.values()], bins=100);


# In[368]:

np.std(test_e_l_results_foo.values())


# In[ ]:



