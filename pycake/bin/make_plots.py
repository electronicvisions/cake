#!/usr/bin/env python
#  -*- coding: utf-8; -*-
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

import argparse
import contextlib
import os
import shutil
import traceback

import numpy as np
import pandas

import matplotlib

# http://stackoverflow.com/a/4935945/1350789
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from pycake.calibration.E_l_I_gl_fixed import E_l_I_gl_fixed_Calibrator
from pycake.helpers.misc import mkdir_p
from pycake.helpers.peakdetect import peakdet
from pycake.reader import Reader
import Coordinate as C
import pycake
import pycake.config
import pycake.helpers.calibtic as calibtic
import pycalibtic
import pyhalbe

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

parser.add_argument("--v_convoffi_runner", help="path to V convoffi runner (if different from 'runner')", default=None)
parser.add_argument("--v_convoffx_runner", help="path to V convoffx runner (if different from 'runner')", default=None)

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

def extract_range_y(reader, config_name, parameter, key, safety_min=0.1, safety_max=0.1):
    """
    extract the range of the measured data for the plots of the calibration
    """
    y = reader.get_results(parameter, reader.get_neurons(), key, repetition = None).values()
    ymin = np.nanmin(y[y != 0]) - safety_min
    ymax = np.nanmax(y) + safety_max
    return ymin, ymax

@contextlib.contextmanager
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
        logger.ERROR("{}: {}".format(msg, e))
        logger.WARN('\n' + traceback.format_exc())


def uncalibrated_hist(xlabel, reader, xscale="linear", yscale="linear", **reader_kwargs):

    if not reader: return

    logger.INFO("uncalibrated hist for {}".format(reader_kwargs["parameter"]))

    reader_kwargs["alpha"] = 0.8

    for include_defects in [True, False]:

        logger.DEBUG("histogram including defects: {}".format(include_defects))

        reader.include_defects = include_defects

        fig, hists = reader.plot_hists(**reader_kwargs)
        plt.legend().set_visible(False)
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
        plt.legend().set_visible(True)

        fig, foos = reader.plot_vs_neuron_number_s(**reader_kwargs)
        plt.title("uncalibrated", x=0.125, y=0.9)
        plt.xlabel("Neuron")
        plt.ylabel(xlabel)
        plt.xlim(0, 512)
        plt.ylim(*reader_kwargs['range'])
        plt.subplots_adjust(**margins)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

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
        plt.legend(loc = 'best')
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

        logger.INFO("... vs neuron number")

        fig, foos = reader.plot_vs_neuron_number_s(**reader_kwargs)
        plt.title("calibrated", x=0.125, y=0.9)
        plt.xlabel("Neuron")
        plt.ylabel(xlabel)
        plt.xlim(0,512)
        plt.subplots_adjust(**margins)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        defects_string = "with_defects" if include_defects else "without_defects"

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "calibrated_vs_neuron_number.pdf"])))

        fig.savefig(os.path.join(fig_dir, "_".join([reader_kwargs["parameter"],
                                                    defects_string,
                                                    "calibrated_vs_neuron_number.png"])))

        #--------------------------------------------------------------------------------

        logger.INFO("... vs shared FG block")

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
        plt.xlabel(xlabel if xlabel != None else label.replace(
            "{inout}", "(in) {}".format(in_unit_label)))
        plt.ylabel(ylabel if ylabel != None else label.replace(
            "{inout}", "(out) {}".format(out_unit_label)))

        plt.subplots_adjust(**margins)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

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



def plot_v_convoff(reader, name="V_convoff_calibrated_test", extra_plots=True, title="calibrated"):
    if not reader:
        raise RuntimeError("missing reader for {}".format(name))

    from pyhalbe.HICANN import neuron_parameter

    with LogError("problem with {} plots".format(name)):
        experiment = reader.runner.get_single(name=name).experiment
        data = experiment.get_all_data(
            [neuron_parameter.I_gl,
             neuron_parameter.V_convoffi,
             neuron_parameter.V_convoffx,
             neuron_parameter.E_l,
             neuron_parameter.E_synx,
             'mean'])
        data.sortlevel('neuron', inplace=True)
        plt_name = name.replace("off","off{}_{}")+".{}"
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
                plt.text(I_gl, 1.10,
                         r"${:.0f} \pm {:.0f}$ mV".format(mean * 1000, std * 1000),
                         rotation=35)

            plt.title(title)
            plt.xlabel("$I_{gl}$ [DAC]")
            plt.ylabel("effective resting potential [V]")
            plt.ylim(0.35,1.25)
            plt.xlim(-10,1023)
            plt.subplots_adjust(**margins)
            plt.grid(True)
            plt.title(title, x=0.125, y=0.9)
            plt.savefig(os.path.join(fig_dir, plt_name.format('', defects_name, 'png')))
            plt.savefig(os.path.join(fig_dir, plt_name.format('', defects_name, 'pdf')))

            if extra_plots:

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

    with LogError("problem with offset backend plots"):
        # offset

        fig = plt.figure()

        calibtic_helper = calibtic.Calibtic(
            args.backenddir, C.Wafer(C.Enum(args.wafer)), C.HICANNOnWafer(C.Enum(args.hicann)))

        def get_offset(cal, nrnidx):
            try:
                offset = cal.nc.at(nrnidx).at(pycalibtic.NeuronCalibrationParameters.Calibrations.ReadoutShift).apply(0)
                return offset
            except IndexError:
                logger.WARN("No offset found for Neuron {}. Is the wafer and hicann enum correct? (w{}, h{})".format(nrnidx,args.wafer,args.hicann))
                return 0
            except RuntimeError, e:
                if e.message == "No transformation available at 23":
                    logger.WARN("No offset found for Neuron {}.".format(nrnidx))
                    return 0
                raise

        offsets = [get_offset(calibtic_helper, n) * 1000 for n in xrange(512)]
        plt.hist(offsets, bins=100);
        plt.subplots_adjust(**margins)
        plt.xlabel("offset [mV]")
        plt.ylabel("#")
        plt.subplots_adjust(**margins)
        plt.xlim(-60,60)
        plt.savefig(os.path.join(fig_dir,"analog_readout_offset.pdf"))
        plt.savefig(os.path.join(fig_dir,"analog_readout_offset.png"))

        fig = plt.figure()
        plt.subplots_adjust(**margins)
        plt.xlabel("neuron")
        plt.ylabel("offset [mV]")
        plt.plot(offsets, 'rx')
        plt.xlim(0,512)
        plt.ylim(-60,60)
        plt.savefig(os.path.join(fig_dir,"analog_readout_offset_vs_nrn.pdf"))
        plt.savefig(os.path.join(fig_dir,"analog_readout_offset_vs_nrn.png"))

    def plot_vsyntc_domains(cal, excitatory=True):
        if excitatory:
            parameter = pycalibtic.NeuronCalibrationParameters.Calibrations.V_syntcx
        else:
            parameter = pycalibtic.NeuronCalibrationParameters.Calibrations.V_syntci

        domains = np.zeros((C.NeuronOnHICANN.enum_type.size, 2))
        for ii, nrn in enumerate(C.iter_all(C.NeuronOnHICANN)):
            try:
                lookup_trafo = cal.nc.at(nrn.id().value()).at(parameter)
            except RuntimeError:
                continue
            bounds = lookup_trafo.getDomainBoundaries()
            low, high = bounds.first, bounds.second
            if low < -1e10 or high > 1e10:
                # We do not care about the old default bounds... Recalculate.
                low_dac = lookup_trafo.apply(low)
                high_dac = lookup_trafo.apply(high)

                low = lookup_trafo.reverseApply(low_dac)
                high = lookup_trafo.reverseApply(high_dac + 1)
            domains[ii] = low, high

        diffs = domains[:, 1] - domains[:, 0]
        indices = diffs.argsort()[::-1]
        domains = domains[indices]
        diffs = diffs[indices]
        means = domains.mean(axis=1)

        fig, ax = plt.subplots(nrows=1)
        fig.subplots_adjust(**margins)
        latex = r"domain of $\tau_{\mathrm{syn}\,%s}$" % (["i", "x"][excitatory])
        ax.set_ylabel("%s [s]" % latex)
        ax.errorbar(
            x=np.arange(0, 512), y=means, yerr=diffs / 2,
            linestyle="", capsize=0., elinewidth=1.,
        )
        ax.set_xlim(0, 512)
        filename = "V_syntc{}_domain".format(["i", "x"][excitatory])
        fig.savefig(os.path.join(fig_dir, "{}.pdf".format(filename)))
        fig.savefig(os.path.join(fig_dir, "{}.png".format(filename)))

    with LogError("problem with V_syntcx domain plots"):
        plot_vsyntc_domains(calibtic_helper, excitatory=True)

    with LogError("problem with V_syntci domain plots"):
        plot_vsyntc_domains(calibtic_helper, excitatory=False)

    with LogError("problem with V_convoff backend plots"):
        def get_vconvoff(cal, nrnidx):
            try:
                convoffx = cal.nc.at(nrnidx).at(pycalibtic.NeuronCalibrationParameters.Calibrations.V_convoffx).apply(0)
                convoffi = cal.nc.at(nrnidx).at(pycalibtic.NeuronCalibrationParameters.Calibrations.V_convoffi).apply(0)
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
        convoffx_l, convoffi_l = zip(*[get_vconvoff(calibtic_helper, n)  for n in xrange(512)])
        bins = np.linspace(0, 1024, 128)
        plt.hist(convoffx_l, label="$V_{convoffx}$", alpha=.5, bins=bins, color="r",
                 range=(np.nanmin(convoffx_l), np.nanmax(convoffx_l)))
        plt.hist(convoffi_l, label="$V_{convoffi}$", alpha=.5, bins=bins, color="b",
                 range=(np.nanmin(convoffi_l), np.nanmax(convoffi_l)))
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

with LogError("problem with V_convoff test plots"):
    r_test_v_convoff = test_reader if args.v_convoff_testrunner == None else Reader(args.v_convoff_testrunner)

    with LogError("problem with uncalibrated V_convoff test plots"):
        plot_v_convoff(r_test_v_convoff, name="V_convoff_test_uncalibrated", extra_plots=False, title="uncalibrated")

    with LogError("problem with calibrated V_convoff test plots"):
        plot_v_convoff(r_test_v_convoff, name="V_convoff_test_calibrated", extra_plots=True, title="calibrated")

## V_convoffi
with LogError("problem with uncalibrated V_convoffi plots"):

    r_v_convoffi = reader if args.v_convoffi_runner == None else Reader(args.v_convoffi_runner)

    if r_v_convoffi:

        unit = r_v_convoffi.runner.get_single(name="V_convoffi")
        calibrator = unit.get_calibrator()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for nrn in args.neuron_enum:

            calibrator.plot_fit_for_neuron(C.NeuronOnHICANN(C.Enum(nrn)),ax)
            plt.ylim(0,0.4)
            plt.xlabel("V_convoffi [DAC] / 1023")

        plt.subplots_adjust(**margins)

        plt.savefig(os.path.join(fig_dir,"V_convoffi_nrns.pdf"))
        plt.savefig(os.path.join(fig_dir,"V_convoffi_nrns.png"))

## V_convoffx
with LogError("problem with uncalibrated V_convoffx plots"):

    r_v_convoffx = reader if args.v_convoffx_runner == None else Reader(args.v_convoffx_runner)

    if r_v_convoffx:

        unit = r_v_convoffx.runner.get_single(name="V_convoffx")
        calibrator = unit.get_calibrator()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for nrn in args.neuron_enum:

            calibrator.plot_fit_for_neuron(C.NeuronOnHICANN(C.Enum(nrn)),ax)
            plt.ylim(0,0.4)
            plt.xlabel("V_convoffx [DAC] / 1023")

        plt.subplots_adjust(**margins)

        plt.savefig(os.path.join(fig_dir,"V_convoffx_nrns.pdf"))
        plt.savefig(os.path.join(fig_dir,"V_convoffx_nrns.png"))

## defects

try:

    r_defects = reader if args.defect_runner == None else Reader(args.defect_runner)

    if r_defects:

        fig, p = r_defects.plot_defect_neurons(sort_by_shared_FG_block=False)
        plt.subplots_adjust(**margins)

        fig.savefig(os.path.join(fig_dir, "defect_neurons_vs_neuron_number.pdf"))
        fig.savefig(os.path.join(fig_dir, "defect_neurons_vs_neuron_number.png"))

        fig, p = r_defects.plot_defect_neurons(sort_by_shared_FG_block=True)
        plt.subplots_adjust(**margins)

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


#  ——— Spikes ——————————————————————————————————————————————————————————————————

try:
    r_test_spikes = test_reader if args.spikes_testrunner == None else Reader(args.spikes_testrunner)

    if r_test_spikes:

        for include_defects in [True, False]:

            defects_string = "with_defects" if include_defects else "without_defects"

            r_test_spikes.include_defects = include_defects

            fig = plt.figure()

            n_spikes = np.array(r_test_spikes.get_results("Spikes",r_test_spikes.get_neurons(), "spikes_n_spikes").values())

            # assume the same E_l for all steps
            E_l = r_test_spikes.runner.config.copy("Spikes").get_parameters()[pyhalbe.HICANN.neuron_parameter.E_l].value

            V_ts = [v[pyhalbe.HICANN.neuron_parameter.V_t].value for v in r_test_spikes.runner.config.copy("Spikes").get_steps()]

            plt.plot(V_ts, np.sum(np.greater(n_spikes,1), axis=0), label="measured")
            #plt.plot(V_ts, np.array(n_spikes_est), label="estimated")

            n_good_neurons = len(r_test_spikes.get_neurons())
            plt.axhline(n_good_neurons, color='black', linestyle="dotted")
            plt.text(min(V_ts)+0.001, n_good_neurons+5, "#not blacklisted", fontsize=12)

            plt.axvline(E_l, color='black', linestyle="dotted")
            plt.text(E_l+0.005,25,'E$_l$', fontsize=12)

            plt.legend(loc="lower left")
            plt.ylabel("# spiking neurons")
            plt.xlabel("$V_t$ [V]")
            plt.xlim(min(V_ts), max(V_ts))

            plt.subplots_adjust(**margins)
            plt.savefig(os.path.join(fig_dir,"n_spiking_neurons_"+defects_string+"_calibrated.png"))

            # number of spikes

            r_test_spikes.include_defects = include_defects

            fig = r_test_spikes.plot_result("Spikes","spikes_n_spikes",yfactor=1,average=True,mark_top_bottom=True)
            plt.legend(loc="lower left")
            plt.ylabel("average number of recorded spikes per neuron")
            plt.xlabel("$V_t$ [DAC]")

            plt.subplots_adjust(**margins)

            defects_string = "with_defects" if include_defects else "without_defects"

            plt.savefig(os.path.join(fig_dir,"_".join(["n_spikes",
                                                       defects_string,
                                                       "result_calibrated.pdf"])))

            plt.savefig(os.path.join(fig_dir,"_".join(["n_spikes",
                                                       defects_string,
                                                       "result_calibrated.png"])))
except Exception as e:
    logger.ERROR("problem with Spikes plots: {}".format(e))

## tau ref

try:
    r_tau_ref = reader if args.tau_ref_runner == None else Reader(args.tau_ref_runner)

    if r_tau_ref:
        xmin, xmax = extract_range_y(r_tau_ref, "I_pl", pyhalbe.HICANN.neuron_parameter.I_pl.name, "tau_ref",  safety_min=0, safety_max=0)

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

        xmin, xmax = extract_range(r_tau_m, "I_gl_PSP", pyhalbe.HICANN.neuron_parameter.I_gl, safety_min=0, safety_max=0)

        xmin /= 10
        xmax *= 10

        uncalibrated_hist(r"$\tau_{m}$ [s]",
                          r_tau_m,
                          xscale="log",
                          parameter="I_gl_PSP",
                          key="tau_2",
                          bins=np.logspace(np.log10(xmin), np.log10(xmax), 100),
                          range=(xmin, xmax),
                          show_legend=True)

        result(r"$\tau_{{m}}$ {inout}", reader=r_tau_m, parameter="I_gl_PSP", key="tau_2", alpha=0.05,
               out_unit_label="[s]")

        #trace("$V_{mem}$ [V]", r_tau_m, "tau_m", args.neuron_enum, end=510, suffix="_uncalibrated")
except Exception as e:
    logger.ERROR("problem with uncalibrated tau_m plots: {}".format(e))

try:
    r_test_tau_m = test_reader if args.tau_m_testrunner == None else Reader(args.tau_m_testrunner)

    if  r_test_tau_m:

        xmin, xmax = extract_range(r_test_tau_m, "I_gl_PSP", pyhalbe.HICANN.neuron_parameter.I_gl, safety_min=0, safety_max=0)

        xmin /= 10
        xmax *= 10

        calibrated_hist(r"$\tau_{m}$ [s]",
                        r_test_tau_m,
                        xscale="log",
                        parameter="I_gl_PSP",
                        key="tau_2",
                        bins=np.logspace(np.log10(xmin), np.log10(xmax), 100),
                        range=(xmin, xmax),
                        show_legend=True)

        result(r"$\tau_{{m}}$ {inout}", reader=r_test_tau_m, suffix="_calibrated", parameter="I_gl_PSP", key="tau_2", alpha=0.05,
               in_unit_label="[s]", out_unit_label="[s]")

        #trace("$V_{mem}$ [V]", r_test_tau_m, parameter="tau_m", neuron=args.neuron_enum, start=500, end=700, suffix="_calibrated")
except Exception as e:
    logger.ERROR("problem with calibrated tau_m plots: {}".format(e))

#  ——— V_syntcx ————————————————————————————————————————————————————————————————

# Mhhh.  We need to provide the smaller of the two time constants.  Patch, patch, patch.
# TODO: PSPAnalyzer should already sort the time constants s.t. tau_1 <= tau_2

@contextlib.contextmanager
def patched_reader_value(reader, parameter, input_keys, *output):
    data = []
    neurons = reader.get_neurons()
    for key_ in input_keys:
        data.append(pandas.DataFrame.from_dict(reader.get_results(parameter, neurons, key_), orient='index').stack())
    results = pandas.concat(data, axis=1)
    results.columns = input_keys
    results.index.names = ["neuron", "step"]
    for key, extract in output:
        results[key] = extract(results)

    try:
        idx = pandas.Index([ids[0].toSharedFGBlockOnHICANN() for ids in results.index.values], name='shared_block')
        results.set_index(idx, append=True, inplace=True)
        results = results.swaplevel('step', 'shared_block').sortlevel('neuron')
        cu = reader.get_calibration_unit(parameter)
        ex_res = cu.experiment.results.sortlevel('neuron')
        cu.experiment.results = pandas.concat([ex_res, results], axis=1)
        yield
    finally:
        if reader.calibration_unit_cache.pop((parameter, 0), None) is None:
            reader.calibration_unit_cache.clear()

def plot_v_syntc(reader, excitatory=True, calibrated=True):
    if not reader:
        return
    if not isinstance(reader, Reader):
        reader = Reader(reader)

    parameter = "V_syntcx" if excitatory else "V_syntci"

    with patched_reader_value(
            reader, parameter, ["tau_1", "tau_2"],
            ("tau_syn", lambda res: res[['tau_1', 'tau_2']].min(axis='columns')),
            ("tau_mem", lambda res: res[['tau_1', 'tau_2']].max(axis='columns'))):

        xmin, xmax = extract_range(
            reader, parameter, getattr(pyhalbe.HICANN.neuron_parameter, parameter),
            safety_min=0, safety_max=0)

        xmin /= 10
        xmax *= 10

        latex = r"$\tau_{\mathrm{syn}\,%s}$" % (["i", "x"][excitatory])
        (calibrated_hist if calibrated else uncalibrated_hist)(
            r"%s [s]" % latex,
            reader=reader,
            parameter=parameter,
            key="tau_syn",
            xscale="log",
            bins=np.logspace(np.log10(xmin), np.log10(xmax), 100),
            range=(xmin, xmax),
            show_legend=True)

        result(
            r"%s {inout}" % latex,
            reader=reader,
            parameter=parameter,
            key="tau_syn",
            suffix="_calibrated" if calibrated else "_uncalibrated",
            alpha=0.05,
            out_unit_label="[s]")

        result(
            r"%s {inout}" % latex,
            ylabel=r"$\tau_{\mathrm{mem}\,%s}$ (out) [s]" % (["i", "x"][excitatory]),
            reader=reader,
            parameter=parameter,
            key="tau_mem",
            suffix="_tau_mem_calibrated" if calibrated else "_tau_mem_uncalibrated",
            alpha=0.05,
            out_unit_label="[s]")

with LogError("problem with uncalibrated v_syntcx plots"):
    plot_v_syntc(args.v_syntcx_runner or reader, excitatory=True, calibrated=False)

with LogError("problem with calibrated v_syntcx plots"):
    plot_v_syntc(args.v_syntcx_testrunner or test_reader, excitatory=True, calibrated=True)

with LogError("problem with uncalibrated v_syntci plots"):
    plot_v_syntc(args.v_syntci_runner or reader, excitatory=False, calibrated=False)

with LogError("problem with calibrated v_syntci plots"):
    plot_v_syntc(args.v_syntci_testrunner or test_reader, excitatory=False, calibrated=True)

cakebin = os.path.split(os.path.abspath(__file__))[0]
shutil.copy(os.path.join(cakebin, "overview.html"), fig_dir)
