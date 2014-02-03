#!/usr/bin/env python
"""Runs E_l calibration, plots and saves data"""

import shutil
import sys
import os
import imp
import argparse

import pycalibtic
import pylogging

# Activate logger before importing other stuff that might want to log
default_logger = pylogging.get("Default")
logger = pylogging.get("run_calibration")
pylogging.default_config()

from pycake.helpers.calibtic import init_backend as init_calibtic
from pycake.helpers.redman import init_backend as init_redman
from pycake.helpers.sthal import StHALContainer
from pyhalbe.HICANN import neuron_parameter, shared_parameter
from pycake.calibration import base, lif, synapse


def check_file(string):
    if not os.path.isfile(string):
        msg = "parameter file '%r' not found! :p" % string
        raise argparse.ArgumentTypeError(msg)
    return string

parser = argparse.ArgumentParser(description='HICANN Calibration tool')
parser.add_argument('parameter_file', type=check_file, nargs='?',
                           help='')
args = parser.parse_args()
parameters = imp.load_source('parameters', args.parameter_file).parameters

neurons = range(512)


# Create necessary folders if the do not exist already
if parameters['save_results']:
    if not os.path.exists(parameters['folder']):
        os.makedirs(parameters['folder'])
        print ("Creating folder {}".format(parameters['folder']))
    if not os.path.exists(parameters['backend_c']):
        os.makedirs(parameters['backend_c'])
        print ("Creating folder {}".format(parameters['backend_c']))
    if not os.path.exists(parameters['backend_r']):
        os.makedirs(parameters['backend_r'])
        print ("Creating folder {}".format(parameters['backend_r']))

coord_wafer  = parameters["coord_wafer"]
coord_hicann = parameters["coord_hicann"]

if parameters["clear"]:
    logger.INFO("Clearing all calibration data.")
    filename_c = os.path.join(parameters["backend_c"], '{}.xml'.format("w{}-h{}".format(int(coord_wafer.value()), int(coord_hicann.id().value()))))
    filename_r = os.path.join(parameters["backend_r"], '{}.xml'.format("hicann-Wafer({})-Enum({})".format(int(coord_wafer.value()), int(coord_hicann.id().value()))))
    if os.path.isfile(filename_c): os.remove(filename_c)
    if os.path.isfile(filename_r): os.remove(filename_r)

# Initialize Calibtic and Redman backends
backend_c = init_calibtic(path=parameters["backend_c"])
backend_r = init_redman(path=parameters["backend_r"])

sthal = StHALContainer(coord_wafer, coord_hicann)


def check_for_existing_calbration(parameter):
    path = parameters['backend_c']
    lib = pycalibtic.loadLibrary('libcalibtic_xml.so')
    backend = pycalibtic.loadBackend(lib)
    backend.config('path', path)
    backend.init()
    md = pycalibtic.MetaData()
    hc = pycalibtic.HICANNCollection()
    if type(parameter) is neuron_parameter:
        try:
            backend.load("w{}-h{}".format(parameters['coord_wafer'].value(), parameters['coord_hicann'].id().value()), md, hc)
            nc = hc.atNeuronCollection()
        except RuntimeError:
            default_logger.INFO('Backend not found. Creating new Backend.')
            return False
        for neuron in range(512):
            try:
                calib = nc.at(neuron).at(parameter)
                default_logger.INFO('Calibration for {} existing.'.format(parameter.name))
                return True
            except (RuntimeError, IndexError):
                continue
            default_logger.INFO('Calibration for {} not existing.'.format(parameter.name))
            return False
    else:
        try:
            backend.load("w{}-h{}".format(parameters['coord_wafer'].value(), parameters['coord_hicann'].id().value()), md, hc)
            bc = hc.atBlockCollection()
        except RuntimeError:
            default_logger.INFO('Backend not found. Creating new Backend.')
            return False
        for block in range(4):
            try:
                calib = bc.at(block).at(parameter)
                default_logger.INFO('Calibration for {} existing.'.format(parameter.name))
                return True
            except (RuntimeError, IndexError):
                continue
            default_logger.INFO('Calibration for {} not existing.'.format(parameter.name))
            return False

pylogging.set_loglevel(default_logger, pylogging.LogLevel.ERROR)

def do_calibration(calibration):
    target_parameter = calibration.target_parameter
    parameter_name = target_parameter.name

    run = parameters["run_" + parameter_name]
    overwrite = parameters['overwrite']
    has_calibration = check_for_existing_calbration(target_parameter)

    if not run:
        return
    if has_calibration and not overwrite:
        logger.INFO("{} already calibrated. Calibration skipped.".format(parameter_name))
        return

    if has_calibration:
        logger.WARN('Overwriting calibration for {}'.format(parameter_name))

    # TODO check from here (and also above ;) )
    calib = calibration(neurons, sthal, parameters)
    pylogging.set_loglevel(calib.logger, pylogging.LogLevel.INFO)
    try:
        calib.run_experiment()
    except Exception,e:
        print "ERROR: ", e
        delete = raw_input("Delete folder {}? (yes / no)".format(calib.folder))
        if delete in ("yes","Yes","y","Y"):
            try:
                shutil.rmtree(calib.folder)
            except OSError as e: # Folder missing, TODO log something
                print e
                pass
        raise

if parameters["calibrate"]:
    for calibration in [synapse.Calibrate_E_synx, synapse.Calibrate_E_syni,
                        lif.Calibrate_E_l, lif.Calibrate_V_t, lif.Calibrate_V_reset, 
                        lif.Calibrate_V_reset_shift, lif.Calibrate_g_l]:
        do_calibration(calibration)

if parameters["measure"]:
    for calibration in [synapse.Test_E_synx, synapse.Test_E_syni,
                        lif.Test_E_l, lif.Test_V_t, lif.Test_V_reset, 
                        lif.Test_g_l]:
        do_calibration(calibration)

quit()

