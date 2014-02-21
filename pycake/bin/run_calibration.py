#!/usr/bin/env python

import shutil
import sys
import os
import imp
import argparse

import pycalibtic
import pylogging

def check_file(string):
    if not os.path.isfile(string):
        msg = "parameter file '%r' not found! :p" % string
        raise argparse.ArgumentTypeError(msg)
    return string

parser = argparse.ArgumentParser(description='HICANN Calibration tool. Takes a parameter file as input. See pycake/bin/parameters.py to see an example.')
parser.add_argument('parameter_file', type=check_file, help='')
args = parser.parse_args()

# Activate logger before importing other stuff that might want to log
default_logger = pylogging.get("Default")
logger = pylogging.get("run_calibration")
pylogging.default_config()

from pycake.helpers.calibtic import init_backend as init_calibtic
from pycake.helpers.redman import init_backend as init_redman
from pycake.helpers.sthal import StHALContainer
from pyhalbe.HICANN import neuron_parameter, shared_parameter
from pycake.calibration import base, lif, synapse


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
            logger.INFO('Backend not found. Creating new Backend.')
            return False
        for neuron in range(512):
            try:
                calib = nc.at(neuron).at(parameter)
                logger.INFO('Calibration for {} existing.'.format(parameter.name))
                return True
            except (RuntimeError, IndexError):
                continue
            logger.INFO('Calibration for {} not existing.'.format(parameter.name))
            return False
    else:
        try:
            backend.load("w{}-h{}".format(parameters['coord_wafer'].value(), parameters['coord_hicann'].id().value()), md, hc)
            bc = hc.atBlockCollection()
        except RuntimeError:
            logger.INFO('Backend not found. Creating new Backend.')
            return False
        for block in range(4):
            try:
                calib = bc.at(block).at(parameter)
                logger.INFO('Calibration for {} existing.'.format(parameter.name))
                return True
            except (RuntimeError, IndexError):
                continue
            logger.INFO('Calibration for {} not existing.'.format(parameter.name))
            return False


def do_calibration(Calibration):
    target_parameter = Calibration.target_parameter
    parameter_name = target_parameter.name

    run = parameters["run_" + parameter_name]
    overwrite = parameters['overwrite']
    has_calibration = check_for_existing_calbration(target_parameter)

    if not run:
        return
    if has_calibration and not overwrite and not issubclass(Calibration, base.BaseTest):
        logger.INFO("{} already calibrated. Calibration skipped.".format(parameter_name))
        return

    if has_calibration and not issubclass(calibration, base.BaseTest):
        logger.WARN('Overwriting calibration for {}'.format(parameter_name))

    # TODO check from here (and also above ;) )
    calib = Calibration(neurons, sthal, parameters)
    pylogging.set_loglevel(calib.logger, pylogging.LogLevel.INFO)
    try:
        # Try several times in case experiment should fail
        for attempt in range(parameters["max_tries"]):
            try:
                calib.run_experiment()
                return
            except RuntimeError, e:
                logger.ERROR(e)
                logger.WARN("Restarting experiment. Try no. {}/{}".format(attempt+1, parameters["max_tries"]))
        raise

    # If all attemps failed:
    except Exception,e:
        folder = getattr(calib, "folder", None)
        if folder and os.path.exists(calib.folder):
            delete = raw_input("Delete folder {}? (yes / no)".format(folder))
            if delete in ("yes","Yes","y","Y"):
                try:
                    shutil.rmtree(folder)
                except OSError as e: # Folder missing, TODO log something
                    print e
                    pass
        raise


logger.INFO("Measuring readout shift.")

#calib_readout_shift = lif.Calibrate_readout_shift(neurons, sthal, parameters)
#calib_readout_shift.run_experiment()

pylogging.set_loglevel(default_logger, pylogging.LogLevel.ERROR)

if parameters["calibrate"]:
    for calibration in [synapse.Calibrate_E_synx, synapse.Calibrate_E_syni,
                        lif.Calibrate_E_l, lif.Calibrate_V_t, lif.Calibrate_V_reset, 
                        lif.Calibrate_I_gl]:
            do_calibration(calibration)


if parameters["measure"]:
    for calibration in [synapse.Test_E_synx, synapse.Test_E_syni,
            lif.Test_E_l, lif.Test_V_t, lif.Test_V_reset, lif.Test_I_gl]:
            do_calibration(calibration)

quit()

