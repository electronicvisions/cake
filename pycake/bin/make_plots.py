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
import pycake.config
from pycake.reader import Reader
from pycake.helpers.peakdetect import peakdet
import pycake.helpers.calibtic as calibtic
import Coordinate as C
import pyhalbe
import pycalibtic
from pycake.calibration.E_l_I_gl_fixed import E_l_I_gl_fixed_Calibrator
import numpy as np
import os
from collections import defaultdict
import argparse
from pycake.helpers.misc import mkdir_p
import shutil
import traceback
from contextlib import contextmanager

import pylogging
pylogging.reset()
pylogging.default_config(date_format='absolute')

logger = pylogging.get("pycake.make_plots")

pylogging.set_loglevel(pylogging.get("pycake.make_plots"), pylogging.LogLevel.TRACE)
pylogging.set_loglevel(pylogging.get("pycake.reader"), pylogging.LogLevel.TRACE)
pylogging.set_loglevel(pylogging.get("pycake.calibrationrunner"), pylogging.LogLevel.TRACE)

font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 12}
matplotlib.rc('font', **font)
margins={"left":0.2, "right":0.95, "top":0.95, "bottom":0.08}
xlog_margins={"left":0.2, "right":0.95, "top":0.95, "bottom":0.1}

parser = argparse.ArgumentParser()
parser.add_argument("runner", help="path of calibration runner directory (can be empty)")
parser.add_argument("testrunner", help="path of test runner directory (evaluation of calibration) (can be empty)")
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

parser.add_argument("--tau_ref_runner", help="path to tau ref runner (if different from 'runner')", default=None)
parser.add_argument("--tau_ref_testrunner", help="path to tau ref test runner (if different from 'testrunner')", default=None)

parser.add_argument("--tau_m_runner", help="path to tau m runner (if different from 'runner')", default=None)
parser.add_argument("--tau_m_testrunner", help="path to tau m test runner (if different from 'testrunner')", default=None)

parser.add_argument("--spikes_testrunner", help="path to spikes test runner (if different from 'testrunner')", default=None)

parser.add_argument("--neuron_enum", help="neuron(s) used for plots", default=[0], type=int, nargs="+")

parser.add_argument("--v_convoff_testrunner", help="path to V convoff test runner (if different from 'testrunner')", default=None)

parser.add_argument("--defect_runner", help="path to runner from which defect neurons will be plotted", default=None)

args = parser.parse_args()

fig_dir = args.outdir
mkdir_p(fig_dir)

try:
    reader = Reader(args.runner)
except Exception as e:
    logger.WARN("Cannot instantiate reader because: {} in \"{}\"".format(e.message, args.runner))
    logger.WARN("Ok, if other runners are specified.")
    reader = None

try:
    test_reader = Reader(args.testrunner)
except Exception as e:
    logger.WARN("Cannot instantiate test reader because: {} in \"{}\"".format(e.message, args.testrunner))
    logger.WARN("Ok, if other runners are specified.")
    test_reader = None

def extract_range(reader, config_name, parameter, safety_min=0.1, safety_max=0.1):
    """
    extract the range for the plots from the steps of the calibration/evaluation
    """
    cfg = reader.runner.config.copy(config_name)
    steps = [step[parameter].value for step in cfg.get_steps()]
    xmin = min(steps) - safety_min
    xmax = max(steps) + safety_max

    return xmin, xmax

@contextmanager
def LogError(msg):
    """
    Use:
        with LogError("Oh no something terrible happend"):
            feae.aeaf = 132
    to catch all exceptions and log them
    """
    try:
        yield
    except Exception as e:
        logger.ERROR(msg + str(e))
        logger.WARN('\n' + traceback.format_exc())


def uncalibrated_hist(xlabel, reader, xscale="linear", yscale="linear", **reader_kwargs):

    if not reader: return

    logger.INFO("uncalibrated hist for {}".format(reader_kwargs["parameter"]))

    reader_kwargs["alpha"] = 0.8

    for include_defects in [True, False]:

        logger.DEBUG("histogram including defects: {}".format(include_defects))

        reader.include_defects = include_defects

        fig, hists = reader.plot_hists(**reader_kwargs)
        plt.title("uncalibrated", x=0.125, y=0.9)
        plt.xlabel(xlabel)
        plt.ylabel("#")
        if 'range' in reader_kwargs:
            plt.xlim(*reader_kwargs['range'])
        if xscale == "log":
            plt.subplots_adjust(**xlog_margins)
        else:
            plt.subplots_adjust(**margins)
        plt.xscale(xscale)
        plt.yscale(yscale)

        defects_string = "with_defects" if include_defects else "without_defects"

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "uncalibrated.pdf"])))

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "uncalibrated.png"])))

        #--------------------------------------------------------------------------------

        logger.INFO("... vs neuron number")

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

        logger.INFO("... vs shared FG block")

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


def calibrated_hist(xlabel, reader, xscale="linear", yscale="linear", **reader_kwargs):

    if not reader: return

    logger.INFO("calibrated hist for {}".format(reader_kwargs["parameter"]))

    reader_kwargs["alpha"] = 0.8

    for include_defects in [True, False]:

        reader.include_defects = include_defects

        fig, hists = reader.plot_hists(**reader_kwargs)
        plt.title("calibrated", x=0.125, y=0.9)
        plt.xlabel(xlabel)
        plt.ylabel("#")
        if 'range' in reader_kwargs:
            plt.xlim(*reader_kwargs['range'])
        if xscale == "log":
            plt.subplots_adjust(**xlog_margins)
        else:
            plt.subplots_adjust(**margins)
        plt.xscale(xscale)
        plt.yscale(yscale)

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
        plt.title("calibrated", x=0.125, y=0.9)
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


def trace(ylabel, reader, parameter, neuron_enum, steps=None, start=0, end=-1, suffix=""):
    """
    neuron: neuron enums
    """

    if not reader: return

    logger.INFO("trace for {}".format(parameter))

    recurrence = 0
    e = reader.get_calibration_unit(name=parameter, recurrence=recurrence).experiment

    if steps == None:
        steps = range(len(e.measurements))

    for n_e in neuron_enum:

        logger.INFO("\t neuron {}".format(n_e))

        neuron = C.NeuronOnHICANN(C.Enum(n_e))

        fig = plt.figure()

        t = None

        for step in steps:

            try:
                p, t = reader.plot_trace(parameter, neuron, step, start, end)
            except KeyError as e:
                logger.WARN(e)
                continue

        plt.grid(True)
        if t:
            plt.xlim(t[0][0], t[0][-1])
        plt.xlabel("t [$\mu$s]")
        plt.ylabel(ylabel)
        plt.subplots_adjust(**margins)
        plt.savefig(os.path.join(fig_dir,parameter+"_trace"+suffix+"_nrn_"+str(neuron.id().value())+".pdf"))
        plt.savefig(os.path.join(fig_dir,parameter+"_trace"+suffix+"_nrn_"+str(neuron.id().value())+".png"))

def result(label, xlabel=None, ylabel=None, reader=None, suffix="",
           xlim=None, ylim=None,
           in_unit_label="[DAC]", out_unit_label="[V]", **reader_kwargs):
    """ label must have placeholder 'inout' for 'in' and 'out' x and y labels,
        like: '$E_{{synx}}$ {inout}'
    """

    if not reader: return

    logger.INFO("result for {}".format(reader_kwargs["parameter"]))

    for include_defects in [True, False]:

        reader.include_defects = include_defects

        fig, plot = reader.plot_result(**reader_kwargs)
        plt.xlabel(xlabel if xlabel != None else label.format(inout="(in) {}".format(in_unit_label)))
        plt.ylabel(ylabel if ylabel != None else label.format(inout="(out) {}".format(out_unit_label)))

        plt.subplots_adjust(**margins)

        if xlim:
            plt.xlim(*xlim)

        if ylim:
            plt.ylim(*ylim)

        plt.grid(True)

        defects_string = "with_defects" if include_defects else "without_defects"

        plt.savefig(os.path.join(fig_dir,"_".join([reader_kwargs["parameter"],
                                                   defects_string,
                                                   "result"+suffix+".pdf"])))

        plt.savefig(os.path.join(fig_dir,"_".join([reader_kwargs["parameter"],
                                                   defects_string,
                                                   "result"+suffix+".png"])))



def plot_v_convoff(reader):
    if not reader:
        return

    from pyhalbe.HICANN import neuron_parameter

    with LogError("problem with uncalibrated V_convoff_test plots"):
        experiment = reader.runner.get_single(name="V_convoff_test").experiment
        data = experiment.get_all_data(
            (neuron_parameter.I_gl,
             neuron_parameter.V_convoffi,
             neuron_parameter.V_convoffx,
             neuron_parameter.E_l,
             neuron_parameter.E_synx),
            ('mean',))
        plt_name = "V_convoff{}_{}_calibrated.{}"
        for include_defects in [True, False]:
            reader.include_defects = include_defects
            nrns = reader.get_neurons() # TODO ???
            defects_name = "with_defects" if include_defects else "without_defects"

            fig = plt.figure()
            for nrn, nrndata in data.loc[nrns].groupby(level='neuron'):
                mean = nrndata.groupby('I_gl').mean()
                std = nrndata.groupby('I_gl').std()
                #ax.errorbar(mean.index, mean.values, yerr=std.values, alpha=0.3, color='k')
                plt.plot(nrndata['I_gl'], nrndata['mean'], 'x', alpha=0.1, color='k')
            for I_gl, tmpdata in data.loc[nrns].groupby('I_gl'):
                mean = tmpdata['mean'].mean()
                std = tmpdata['mean'].std()
                plt.text(I_gl, 0.6,
                         r"$\sigma = {:.0f}mV$".format(std * 1000),
                         rotation='vertical')

            title = "$E_l$ variation after offset calibration"
            if not include_defects:
                title += " (without {} defect neurons)".format(512 - len(nrns))
            plt.title(title)
            plt.xlabel("$I_{gl} [DAC]$")
            plt.ylabel("effective resting potential [V]")
            plt.subplots_adjust(**margins)
            plt.grid(True)
            plt.savefig(os.path.join(fig_dir, plt_name.format('', defects_name, 'png')))
            plt.savefig(os.path.join(fig_dir, plt_name.format('', defects_name, 'pdf')))

            hist_data = data.loc[nrns].xs(0, level='step')

            args = {"bins": np.linspace(0, 1023, 100)}

            fig = plt.figure()
            plt.hist(data['V_convoffi'].values, **args)
            plt.xlabel("choosen $V_{convoffi}$ [DAC]")
            plt.ylabel("effective resting potential [V]")
            plt.subplots_adjust(**margins)
            plt.grid(True)
            plt.savefig(os.path.join(fig_dir, plt_name.format('i', defects_name, 'png')))
            plt.savefig(os.path.join(fig_dir, plt_name.format('i', defects_name, 'pdf')))

            fig = plt.figure()
            plt.hist(data['V_convoffx'].values, **args)
            plt.xlabel("choosen $V_{convoffx}$ [DAC]")
            plt.ylabel("effective resting potential [V]")
            plt.subplots_adjust(**margins)
            plt.grid(True)
            plt.savefig(os.path.join(fig_dir, plt_name.format('x', defects_name, 'png')))
            plt.savefig(os.path.join(fig_dir, plt_name.format('x', defects_name, 'pdf')))

            fig = plt.figure()
            for I_gl, nrndata in data.loc[nrns].groupby("I_gl"):
                too_high_or_low = nrndata[abs(nrndata['mean'] - nrndata['mean'].mean()) > nrndata['mean'].std()]
                plt.hist(too_high_or_low['V_convoffx'], bins=np.linspace(0, 1023, 100), label="I_gl={} DAC".format(I_gl), alpha=0.5)
            plt.legend(loc=0)
            plt.xlabel("choosen $V_{convoffx}$ [DAC]")
            plt.ylabel("#")
            #plt.subplots_adjust(**margins)
            plt.savefig(os.path.join(fig_dir, plt_name.format('_V_convoffx_E_l_too_high_or_low', defects_name, 'png')))
            plt.savefig(os.path.join(fig_dir, plt_name.format('_V_convoffx_E_l_too_high_or_low', defects_name, 'pdf')))

if args.backenddir:

    # offset

    fig = plt.figure()

    c = calibtic.Calibtic(args.backenddir,C.Wafer(C.Enum(args.wafer)),C.HICANNOnWafer(C.Enum(args.hicann)))

    def get_offset(cal, nrnidx):
        try:
            offset = cal.nc.at(nrnidx).at(pycalibtic.NeuronCalibration.ReadoutShift).apply(0)
            return offset
        except IndexError:
            logger.WARN("No offset found for Neuron {}. Is the wafer and hicann enum correct? (w{}, h{})".format(nrnidx,args.wafer,args.hicann))
            return 0
        except RuntimeError, e:
            if e.message == "No transformation available at 23":
                logger.WARN("No offset found for Neuron {}.".format(nrnidx))
                return 0
            raise

    offsets = [get_offset(c, n) * 1000 for n in xrange(512)]
    plt.hist(offsets, bins=100);
    plt.xlabel("offset [mV]")
    plt.ylabel("#")
    plt.subplots_adjust(**margins)
    plt.savefig(os.path.join(fig_dir,"analog_readout_offset.pdf"))
    plt.savefig(os.path.join(fig_dir,"analog_readout_offset.png"))

    fig = plt.figure()
    plt.xlabel("neuron")
    plt.ylabel("offset [mV]")
    plt.plot(offsets, 'rx')
    plt.savefig(os.path.join(fig_dir,"analog_readout_offset_vs_nrn.pdf"))
    plt.savefig(os.path.join(fig_dir,"analog_readout_offset_vs_nrn.png"))

    def get_vconvoff(cal, nrnidx):
        try:
            convoffx = cal.nc.at(nrnidx).at(pyhalbe.HICANN.neuron_parameter.V_convoffx).apply(0)
            convoffi = cal.nc.at(nrnidx).at(pyhalbe.HICANN.neuron_parameter.V_convoffi).apply(0)
        except IndexError as i:
            logger.WARN(str(i) + ", No V_convoff(i or x) found for Neuron {}. Is the wafer and hicann enum correct? (w{}, h{})".format(nrnidx,args.wafer,args.hicann))
            return np.nan, np.nan
        except RuntimeError as e:
            if str(e).startswith("No transformation available at"):
                logger.WARN(str(e) + ", No V_convoff(i or x) found for Neuron {}.".format(nrnidx))
                return np.nan, np.nan
            raise

        return convoffx, convoffi

    fig = plt.figure()
    convoffx_l, convoffi_l = zip(*[get_vconvoff(c, n)  for n in xrange(512)])
    bins = np.linspace(0, 1024, 128)
    plt.hist(convoffx_l, label="$V_{convoffx}$", alpha=.5, bins=bins, color="r");
    plt.hist(convoffi_l, label="$V_{convoffi}$", alpha=.5, bins=bins, color="b");
    plt.xlim(0, 1024)
    plt.legend()
    plt.xlabel("$V_{convoff}$ [DAC]")
    plt.ylabel("#")
    plt.subplots_adjust(**margins)
    plt.savefig(os.path.join(fig_dir,"V_convoff.pdf"))
    plt.savefig(os.path.join(fig_dir,"V_convoff.png"))

    fig = plt.figure()
    plt.xlabel("neuron")
    plt.ylabel("$V_{convoff}$ [DAC]")
    plt.plot(convoffx_l, 'rx', label="$V_{convoffx}$")
    plt.plot(convoffi_l, 'bx', label="$V_{convoffi}$")
    plt.legend()
    plt.xlim(0, 512)
    plt.ylim(0, 1024)
    plt.subplots_adjust(**margins)
    plt.savefig(os.path.join(fig_dir,"V_convoff_vs_nrn.pdf"))
    plt.savefig(os.path.join(fig_dir,"V_convoff_vs_nrn.png"))

## V convoff
try:

    r_test_v_convoff = test_reader if args.v_convoff_testrunner == None else Reader(args.v_convoff_testrunner)

    plot_v_convoff(r_test_v_convoff)

except Exception as e:
    logger.ERROR("problem with V_convoff test plots: {}".format(e))

## defects

try:

    r_defects = reader if args.defect_runner == None else Reader(args.defect_runner)

    fig, p = r_defects.plot_defect_neurons(sort_by_shared_FG_block=False)
    fig.savefig(os.path.join(fig_dir, "defect_neurons_vs_neuron_number.pdf"))
    fig.savefig(os.path.join(fig_dir, "defect_neurons_vs_neuron_number.png"))

    fig, p = r_defects.plot_defect_neurons(sort_by_shared_FG_block=True)
    fig.savefig(os.path.join(fig_dir, "defect_neurons_vs_shared_FG_block.pdf"))
    fig.savefig(os.path.join(fig_dir, "defect_neurons_vs_shared_FG_block.png"))

except Exception as e:
    logger.ERROR("problem with neuron defect plots: {}".format(e))

## V reset
try:
    r_v_reset = reader if args.v_reset_runner == None else Reader(args.v_reset_runner)

    if r_v_reset:

        xmin, xmax = extract_range(r_v_reset, "V_reset", pyhalbe.HICANN.shared_parameter.V_reset)

        uncalibrated_hist("$V_{reset}$ [V]",
                          r_v_reset,
                          parameter="V_reset",
                          key="baseline",
                          bins=100,
                          range=(xmin, xmax),
                          show_legend=True)

        trace("$V_{mem}$ [mV]", r_v_reset, "V_reset", args.neuron_enum, end=2000, suffix="_uncalibrated")

        result("$V_{{reset}}$ {inout}", reader=r_v_reset, parameter="V_reset",key="baseline",alpha=0.05,step_key=pyhalbe.HICANN.shared_parameter.V_reset)
except Exception as e:
    logger.ERROR("problem with uncalibrated V_reset plots: {}".format(e))

try:
    r_test_v_reset = test_reader if args.v_reset_testrunner == None else Reader(args.v_reset_testrunner)

    if r_test_v_reset:

        xmin, xmax = extract_range(r_test_v_reset, "V_reset", pyhalbe.HICANN.shared_parameter.V_reset)

        calibrated_hist("$V_{reset}$ [V]",
                          r_test_v_reset,
                          parameter="V_reset",
                          key="baseline",
                          bins=100,
                          range=(xmin, xmax),
                          show_legend=True)

        trace("$V_{mem}$ [V]", r_test_v_reset, "V_reset", args.neuron_enum, end=510, suffix="_calibrated")
except Exception as e:
    logger.ERROR("problem with calibrated V_reset plots: {}".format(e))

## E synx

try:
    r_e_synx = reader if args.e_synx_runner == None else Reader(args.e_synx_runner)

    if r_e_synx:

        xmin, xmax = extract_range(r_e_synx, "E_synx", pyhalbe.HICANN.neuron_parameter.E_synx)

        uncalibrated_hist("$E_{synx}$ [V]",
                          r_e_synx,
                          parameter="E_synx",
                          key="mean",
                          bins=100,
                          range=(xmin,xmax),
                          show_legend=True);

        result("$E_{{synx}}$ {inout}", reader=r_e_synx, ylim=[xmin,xmax], parameter="E_synx",key="mean",alpha=0.05, step_key=pyhalbe.HICANN.neuron_parameter.E_synx)

        trace("$V_{mem}$ [V]", r_e_synx, "E_synx", args.neuron_enum, end=510, suffix="_uncalibrated")
except Exception as e:
    logger.ERROR("problem with uncalibrated E_synx plots: {}".format(e))

try:
    r_test_e_synx = test_reader if args.e_synx_testrunner == None else Reader(args.e_synx_testrunner)

    if r_test_e_synx:

        xmin, xmax = extract_range(r_test_e_synx, "E_synx", pyhalbe.HICANN.neuron_parameter.E_synx)

        calibrated_hist("$E_{synx}$ [V]",
                        r_test_e_synx,
                        parameter="E_synx",
                        key="mean",
                        bins=100,
                        range=(xmin, xmax),
                        show_legend=True);

        result("$E_{{synx}}$ {inout}", reader=r_test_e_synx, suffix="_calibrated", xlim=[xmin/1.8*1023,xmax/1.8*1023], ylim=[xmin,xmax], parameter="E_synx",key="mean",alpha=0.05, step_key=pyhalbe.HICANN.neuron_parameter.E_synx)

        trace("$V_{mem}$ [V]", r_test_e_synx, "E_synx", args.neuron_enum, end=510, suffix="_calibrated")

    #for k,v in r_test_e_synx.get_results("E_synx", r_test_e_synx.get_neurons(), "mean").iteritems():
    #        if v[0] < 0.78 or v[0] > 0.82:
    #            print k, k.id(), v[0]
except Exception as e:
    logger.ERROR("problem with calibrated E_synx plots: {}".format(e))

## E syni

try:
    r_e_syni = reader if args.e_syni_runner == None else Reader(args.e_syni_runner)

    if r_e_syni:

        xmin, xmax = extract_range(r_e_syni, "E_syni", pyhalbe.HICANN.neuron_parameter.E_syni)

        uncalibrated_hist("$E_{syni}$ [V]",
                          r_e_syni,
                          parameter="E_syni",
                          key="mean",
                          bins=100,
                          range=(xmin, xmax),
                          show_legend=True);

        result("$E_{{syni}}$ {inout}", reader=r_e_syni, ylim=[xmin,xmax], parameter="E_syni",key="mean",alpha=0.05, step_key=pyhalbe.HICANN.neuron_parameter.E_syni)

        trace("$V_{mem}$ [V]", r_e_syni, "E_syni", args.neuron_enum, end=510, suffix="_uncalibrated")
except Exception as e:
    logger.ERROR("problem with uncalibrated E_syni plots: {}".format(e))

try:
    r_test_e_syni = test_reader if args.e_syni_testrunner == None else Reader(args.e_syni_testrunner)

    if r_test_e_syni:

        xmin, xmax = extract_range(r_test_e_syni, "E_syni", pyhalbe.HICANN.neuron_parameter.E_syni)
        
        calibrated_hist("$E_{syni}$ [V]",
                        r_test_e_syni,
                        parameter="E_syni",
                        key="mean",
                        bins=100,
                        range=(xmin, xmax),
                        show_legend=True);

        result("$E_{{syni}}$ {inout}", reader=r_test_e_syni, suffix="_calibrated", xlim=[xmin/1.8*1023,xmax/1.8*1023], ylim=[xmin,xmax], parameter="E_syni",key="mean",alpha=0.05, step_key=pyhalbe.HICANN.neuron_parameter.E_syni)

        trace("$V_{mem}$ [V]", r_test_e_syni, "E_syni", args.neuron_enum, end=510, suffix="_calibrated")
except Exception as e:
    logger.ERROR("problem with calibrated E_syni plots: {}".format(e))

## E l

try:
    r_e_l = reader if args.e_l_runner == None else Reader(args.e_l_runner)

    if r_e_l:

        xmin, xmax = extract_range(r_e_l, "E_l", pyhalbe.HICANN.neuron_parameter.E_l)

        uncalibrated_hist("$E_{l}$ [V]",
                          r_e_l,
                          parameter="E_l",
                          key="mean",
                          bins=100,
                          range=(xmin,xmax),
                          show_legend=True)

        result("$E_{{l}}$ {inout}", reader=r_e_l, ylim=[xmin,xmax], parameter="E_l",key="mean",alpha=0.05, step_key=pyhalbe.HICANN.neuron_parameter.E_l)

        trace("$V_{mem}$ [V]", r_e_l, "E_l", args.neuron_enum, end=510, suffix="_uncalibrated")

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
        plt.ylabel("Output [V]")
        plt.subplots_adjust(**margins)
        plt.xlim(500,900)
        plt.ylim(500,900)
        plt.legend(loc="upper left")
        plt.grid(True)
        fig.savefig(os.path.join(fig_dir,"calib_example_lines.pdf"))
        """
except Exception as e:
    logger.ERROR("problem with uncalibrated E_l plots: {}".format(e))

try:
    r_test_e_l = test_reader if args.e_l_testrunner == None else Reader(args.e_l_testrunner)

    if r_test_e_l:

        xmin, xmax = extract_range(r_test_e_l, "E_l", pyhalbe.HICANN.neuron_parameter.E_l)

        calibrated_hist("$E_{l}$ [V]",
                        r_test_e_l,
                        parameter="E_l",
                        key="mean",
                        show_legend=True,
                        bins=100,
                        range=(xmin,xmax))

        result("$E_{{l}}$ {inout}", reader=r_test_e_l, suffix="_calibrated", xlim=[xmin/1.8*1023,xmax/1.8*1023], ylim=[xmin,xmax], parameter="E_l",key="mean",alpha=0.05, step_key=pyhalbe.HICANN.neuron_parameter.E_l)

        trace("$V_{mem}$ [V]", r_test_e_l, "E_l", args.neuron_enum, end=510, suffix="_calibrated")
except Exception as e:
    logger.ERROR("problem with calibrated E_l plots: {}".format(e))

## V t

try:
    r_v_t = reader if args.v_t_runner == None else Reader(args.v_t_runner)

    if r_v_t:

        xmin, xmax = extract_range(r_v_t, "V_t", pyhalbe.HICANN.neuron_parameter.V_t)

        uncalibrated_hist("$V_{t}$ [V]",
                          r_v_t,
                          parameter="V_t",
                          key="max",
                          bins=100,
                          range=(xmin,xmax),
                          show_legend=True)

        result("$V_{{t}}$ {inout}", reader=r_v_t, xlim=[xmin/1.8*1023,xmax/1.8*1023], ylim=[xmin,xmax], parameter="V_t",key="max",alpha=0.05, step_key=pyhalbe.HICANN.neuron_parameter.V_t)

        trace("$V_{mem}$ [V]", r_v_t, "V_t", args.neuron_enum, end=510, suffix="_uncalibrated")
except Exception as e:
    logger.ERROR("problem with uncalibrated V_t plots: {}".format(e))

try:
    r_test_v_t = test_reader if args.v_t_testrunner == None else Reader(args.v_t_testrunner)

    if  r_test_v_t:

        xmin, xmax = extract_range(r_test_v_t, "V_t", pyhalbe.HICANN.neuron_parameter.V_t)

        calibrated_hist("$V_{t}$ [V]",
                        r_test_v_t,
                        parameter="V_t",
                        key="max",
                        bins=100,
                        range=(xmin,xmax),
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

        result("$V_{{t}}$ {inout}", reader=r_test_v_t, suffix="_calibrated", xlim=[xmin/1.8*1023,xmax/1.8*1023], ylim=[xmin,xmax], parameter="V_t",key="max",alpha=0.05, step_key=pyhalbe.HICANN.neuron_parameter.V_t)

        trace("$V_{mem}$ [V]", r_test_v_t, "V_t", args.neuron_enum, start=500, end=700, suffix="_calibrated")

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
except Exception as e:
    logger.ERROR("problem with calibrated V_t plots: {}".format(e))

## E l, I gl

# In[72]:

#r_e_l_i_gl = #Reader("/afsuser/sschmitt/.calibration-restructure/runner_E_l_I_gl_fixed_0527_1211.p.bz2")


# In[76]:

#e=r_e_l_i_gl.runner.experiments["E_l_I_gl_fixed"]
#m=e.measurements[-1]
#calibrator = E_l_I_gl_fixed_Calibrator(e)
#print calibrator.fit_neuron(C.NeuronOnHICANN(C.Enum(1)))


## V syntcx psp max

try:
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
               mark_top_bottom=True,
               alpha=0.5,
               marker="None",
               yfactor=1000,
               ylim=[0,25])


        trace("$V_{mem}$ [mV]", r_v_syntcx, "V_syntcx_psp_max", args.neuron_enum, end=4000)

        # In[311]:

        fig = plt.figure()
        r_v_syntcx.include_defects = False
        results_v_syntcx = r_v_syntcx.get_results("V_syntcx_psp_max",r_v_syntcx.get_neurons(),"std")

        bins = np.linspace(0,50,101)

        top_max_stds = [np.max(stds)*1000 for n, stds in results_v_syntcx.iteritems() if n.y() == C.Y(0)]
        plt.hist(top_max_stds,bins=bins,color='b',label="top")

        bottom_max_stds = [np.max(stds)*1000 for n, stds in results_v_syntcx.iteritems() if n.y() == C.Y(1)]
        plt.hist(bottom_max_stds,bins=bins,color='g',alpha=0.8,label="bottom")

        plt.xlabel("max $\sigma$(trace) [mV]")
        plt.ylabel("#")
        plt.xlim(0,50)
        plt.ylim(0,23)
        plt.legend()
        plt.subplots_adjust(**margins)
        plt.savefig(os.path.join(fig_dir,"V_syntcx_psp_stds.pdf"))
        plt.savefig(os.path.join(fig_dir,"V_syntcx_psp_stds.png"))

        # In[242]:

        # sum(np.array(max_stds) < 0.005)
except Exception as e:
    logger.ERROR("problem with V_syntcx plots: {}".format(e))

## V syntci psp max

try:
    r_v_syntci = reader if args.v_syntci_runner == None else Reader(args.v_syntci_runner)

    if r_v_syntci:

        e = r_v_syntci.runner.get_single(name="V_syntci_psp_max").experiment

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
               mark_top_bottom=True,
               alpha=0.5,
               marker="None",
               yfactor=1000,
               ylim=[0,25])

        trace("$V_{mem}$ [mV]", r_v_syntci, "V_syntci_psp_max", args.neuron_enum, end=510)

        # In[319]:

        fig = plt.figure()
        r_v_syntci.include_defects = False
        results_v_syntci = r_v_syntci.get_results("V_syntci_psp_max",r_v_syntci.get_neurons(),"std")

        bins = np.linspace(0,50,101)

        top_max_stds = [np.max(stds)*1000 for n, stds in results_v_syntci.iteritems() if n.y() == C.Y(0)]
        plt.hist(top_max_stds,bins=bins,color='b',label="top")

        bottom_max_stds = [np.max(stds)*1000 for n, stds in results_v_syntci.iteritems() if n.y() == C.Y(1)]
        plt.hist(bottom_max_stds,bins=bins,color='g',alpha=0.8,label="bottom")

        plt.xlabel("max $\sigma$(trace) [mV]")
        plt.ylabel("#")
        plt.xlim(0,50)
        plt.ylim(0,23)
        plt.legend()
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
except Exception as e:
    logger.ERROR("problem with V_syntci plots: {}".format(e))

logger.WARN("spike plots disabled until readout is stable")
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
"""

## tau ref

try:
    r_tau_ref = reader if args.tau_ref_runner == None else Reader(args.tau_ref_runner)

    if r_tau_ref:

        xmin, xmax = extract_range(r_tau_ref, "I_pl", pyhalbe.HICANN.neuron_parameter.I_pl, safety_min=0, safety_max=0)

        xmin /= 10
        xmax *= 10

        uncalibrated_hist(r"$\tau_{ref}$ [s]",
                          r_tau_ref,
                          xscale="log",
                          parameter="I_pl",
                          key="tau_ref",
                          bins=np.logspace(np.log10(xmin), np.log10(xmax), 100),
                          range=(xmin, xmax),
                          show_legend=True)

        result(r"$\tau_{{ref}}$ {inout}", reader=r_tau_ref, parameter="I_pl", key="tau_ref", alpha=0.05,
               out_unit_label="[s]")

        #trace("$V_{mem}$ [V]", r_tau_ref, "tau_ref", args.neuron_enum, end=510, suffix="_uncalibrated")

except Exception as e:
    logger.ERROR("problem with uncalibrated tau_ref plots: {}".format(e))

try:
    r_test_tau_ref = test_reader if args.tau_ref_testrunner == None else Reader(args.tau_ref_testrunner)

    if r_test_tau_ref:

        xmin, xmax = extract_range(r_test_tau_ref, "I_pl", pyhalbe.HICANN.neuron_parameter.I_pl, safety_min=0, safety_max=0)

        xmin /= 10
        xmax *= 10

        calibrated_hist(r"$\tau_{ref}$ [s]",
                        r_test_tau_ref,
                        xscale="log",
                        parameter="I_pl",
                        key="tau_ref",
                        bins=np.logspace(np.log10(xmin), np.log10(xmax), 100),
                        range=(xmin, xmax),
                        show_legend=True)

        result(r"$\tau_{{ref}}$ {inout}", reader=r_test_tau_ref, suffix="_calibrated", parameter="I_pl", key="tau_ref", alpha=0.05,
               in_unit_label="[s]", out_unit_label="[s]")

        #trace("$V_{mem}$ [V]", r_test_tau_ref, parameter="tau_ref", args.neuron_enum, start=500, end=700, suffix="_calibrated")

except Exception as e:
    logger.ERROR("problem with calibrated tau_ref plots: {}".format(e))


## tau m
try:
    r_tau_m = reader if args.tau_m_runner == None else Reader(args.tau_m_runner)

    if r_tau_m:

        xmin, xmax = extract_range(r_tau_m, "I_gl", pyhalbe.HICANN.neuron_parameter.I_gl, safety_min=0, safety_max=0)

        xmin /= 10
        xmax *= 10

        uncalibrated_hist(r"$\tau_{m}$ [s]",
                          r_tau_m,
                          xscale="log",
                          parameter="I_gl",
                          key="tau_m",
                          bins=np.logspace(np.log10(xmin), np.log10(xmax), 100),
                          range=(xmin, xmax),
                          show_legend=True)

        result(r"$\tau_{{m}}$ {inout}", reader=r_tau_m, parameter="I_gl", key="tau_m", alpha=0.05,
               out_unit_label="[s]")

        #trace("$V_{mem}$ [V]", r_tau_m, "tau_m", args.neuron_enum, end=510, suffix="_uncalibrated")
except Exception as e:
    logger.ERROR("problem with uncalibrated tau_m plots: {}".format(e))

try:
    r_test_tau_m = test_reader if args.tau_m_testrunner == None else Reader(args.tau_m_testrunner)

    if  r_test_tau_m:

        xmin, xmax = extract_range(r_test_tau_m, "I_gl", pyhalbe.HICANN.neuron_parameter.I_gl, safety_min=0, safety_max=0)

        xmin /= 10
        xmax *= 10

        calibrated_hist(r"$\tau_{m}$ [s]",
                        r_test_tau_m,
                        xscale="log",
                        parameter="I_gl",
                        key="tau_m",
                        bins=np.logspace(np.log10(xmin), np.log10(xmax), 100),
                        range=(xmin, xmax),
                        show_legend=True)

        result(r"$\tau_{{m}}$ {inout}", reader=r_test_tau_m, suffix="_calibrated", parameter="I_gl", key="tau_m", alpha=0.05,
               in_unit_label="[s]", out_unit_label="[s]")

        #trace("$V_{mem}$ [V]", r_test_tau_m, parameter="tau_m", neuron=args.neuron_enum, start=500, end=700, suffix="_calibrated")
except Exception as e:
    logger.ERROR("problem with calibrated tau_m plots: {}".format(e))

cakebin = os.path.split(os.path.abspath(__file__))[0]
shutil.copy(os.path.join(cakebin, "overview.html"), fig_dir)
