"""Runs E_l calibration, plots and saves data"""

from pycake.helpers.calibtic import init_backend as init_calibtic
from pycake.helpers.redman import init_backend as init_redman
from pycake.helpers.sthal import StHALContainer
from pyhalbe.HICANN import neuron_parameter, shared_parameter
from pycake.calibration import base, lif, synapse

import shutil
import os

import pylogging

import sys
import pycalibtic

import imp

# load specified file. if no file given, load standard file
if len(sys.argv)>1 and os.path.isfile(sys.argv[1]):
    parameters = imp.load_source('parameters', sys.argv[1]).parameters
else:
    parameters = imp.load_source('parameters', 'parameters.py').parameters

neurons = range(512)

default_logger = pylogging.get("Default")
pylogging.set_loglevel(default_logger, pylogging.LogLevel.INFO)

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
    print ("Clearing all calibration data")
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

# Start measuring

if parameters["calibrate"]:
    if parameters["run_E_synx"]:
        if parameters['overwrite'] or (not check_for_existing_calbration(neuron_parameter.E_synx)):
            if parameters['overwrite'] and check_for_existing_calbration(neuron_parameter.E_synx):
                print 'Overwriting calibration for E_synx'
            calib_E_synx = synapse.Calibrate_E_synx(neurons, sthal, parameters)
            pylogging.set_loglevel(calib_E_synx.logger, pylogging.LogLevel.INFO)
            try:
                calib_E_synx.run_experiment()
            except Exception,e:
                print "ERROR: ", e
                delete = raw_input("Delete folder {}? (yes / no)".format(calib_E_synx.folder))
                if delete in ("yes","Yes","y","Y"):
                    shutil.rmtree(calib_E_synx.folder)
                raise
        else:
            print "E_synx already calibrated. Calibration skipped."

    if parameters["run_E_syni"]:
        if parameters['overwrite'] or (not check_for_existing_calbration(neuron_parameter.E_syni)):
            if parameters['overwrite'] and check_for_existing_calbration(neuron_parameter.E_syni):
                print 'Overwriting calibration for E_syni'
            calib_E_syni = synapse.Calibrate_E_syni(neurons, sthal, parameters)
            pylogging.set_loglevel(calib_E_syni.logger, pylogging.LogLevel.INFO)
            try:
                calib_E_syni.run_experiment()
            except Exception,e:
                print "ERROR: ", e
                delete = raw_input("Delete folder {}? (yes / no)".format(calib_E_syni.folder))
                if delete in ("yes","Yes","y","Y"):
                    shutil.rmtree(calib_E_syni.folder)
                raise e
        else:
            print "E_syni already calibrated. Calibration skipped."

    if parameters["run_E_l"]:
        if parameters['overwrite'] or (not check_for_existing_calbration(neuron_parameter.E_l)):
            if parameters['overwrite'] and check_for_existing_calbration(neuron_parameter.E_l):
                print 'Overwriting calibration for E_l'
            calib_E_l = lif.Calibrate_E_l(neurons, sthal, parameters)
            pylogging.set_loglevel(calib_E_l.logger, pylogging.LogLevel.INFO)
            try:
                calib_E_l.run_experiment()
            except Exception,e:
                print "ERROR: ", e
                delete = raw_input("Delete folder {}? (yes / no)".format(calib_E_l.folder))
                if delete in ("yes","Yes","y","Y"):
                    shutil.rmtree(calib_E_l.folder)
                raise e
        else:
            print "E_l already calibrated. Calibration skipped."
    
    if parameters["run_V_t"]:
        if parameters['overwrite'] or (not check_for_existing_calbration(neuron_parameter.V_t)):
            if parameters['overwrite'] and check_for_existing_calbration(neuron_parameter.V_t):
                print 'Overwriting calibration for V_t'
            calib_V_t = lif.Calibrate_V_t(neurons, sthal, parameters)
            pylogging.set_loglevel(calib_V_t.logger, pylogging.LogLevel.INFO)
            try:
                calib_V_t.run_experiment()
            except Exception,e:
                print "ERROR: ", e
                delete = raw_input("Delete folder {}? (yes / no)".format(calib_V_t.folder))
                if delete in ("yes","Yes","y","Y"):
                    shutil.rmtree(calib_V_t.folder)
                raise e
        else:
            print "V_t already calibrated. Calibration skipped."
    
    if parameters["run_V_reset"]:
        if parameters['overwrite'] or (not check_for_existing_calbration(shared_parameter.V_reset)):
            if parameters['overwrite'] and check_for_existing_calbration(shared_parameter.V_reset):
                print 'Overwriting calibration for V_reset'
            # Calibrate V_reset
            calib_V_reset = lif.Calibrate_V_reset(neurons, sthal, parameters)
            pylogging.set_loglevel(calib_V_reset.logger, pylogging.LogLevel.INFO)
            try:
                calib_V_reset.run_experiment()
            except Exception,e:
                print "ERROR: ", e
                delete = raw_input("Delete folder {}? (yes / no)".format(calib_V_reset.folder))
                if delete in ("yes","Yes","y","Y"):
                    shutil.rmtree(calib_V_reset.folder)
                    shutil.rmtree(calib_V_reset_shift.folder)
                raise e
            # Calibrate V_reset_shift
            calib_V_reset_shift = lif.Calibrate_V_reset_shift(neurons, sthal, parameters)
            pylogging.set_loglevel(calib_V_reset_shift.logger, pylogging.LogLevel.INFO)
            try:
                calib_V_reset_shift.run_experiment()
            except Exception,e:
                print "ERROR: ", e
                delete = raw_input("Delete folder {}? (yes / no)".format(calib_V_reset.folder))
                if delete in ("yes","Yes","y","Y"):
                    shutil.rmtree(calib_V_reset_shift.folder)
                raise e
        else:
            print "V_reset already calibrated. Calibration skipped."
    
    if parameters["run_I_gl"]:
        calib_I_gl = lif.Calibrate_g_L(neurons, sthal, parameters)
        pylogging.set_loglevel(calib_I_gl.logger, pylogging.LogLevel.INFO)
        #try:
        calib_I_gl.run_experiment()
        #except:
        #    print "ERROR: ", e
        #    delete = raw_input("Delete folder {}? (yes / no)".format(calib_I_gl.folder))
        #    if delete in ("yes","Yes","y","Y"):
        #        shutil.rmtree(calib_I_gl.folder)
        #    raise e


if parameters["measure"]:
    if parameters["run_E_synx"]:
        test_E_synx = synapse.Test_E_synx(neurons, sthal, parameters)
        pylogging.set_loglevel(test_E_synx.logger, pylogging.LogLevel.INFO)
        try:
            test_E_synx.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder {}? (yes / no)".format(test_E_synx.folder))
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(test_E_synx.folder)
            raise e

    if parameters["run_E_syni"]:
        test_E_syni = synapse.Test_E_syni(neurons, sthal, parameters)
        pylogging.set_loglevel(test_E_syni.logger, pylogging.LogLevel.INFO)
        try:
            test_E_syni.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder {}? (yes / no)".format(test_E_syni.folder))
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(test_E_syni.folder)
            raise e

    if parameters["run_E_l"]:
        test_E_l = lif.Test_E_l(neurons, sthal, parameters)
        pylogging.set_loglevel(test_E_l.logger, pylogging.LogLevel.INFO)
        try:
            test_E_l.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder {}? (yes / no)".format(test_E_l.folder))
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(test_E_l.folder)
            raise e
    
    if parameters["run_V_t"]:
        test_V_t = lif.Test_V_t(neurons, sthal, parameters)
        pylogging.set_loglevel(test_V_t.logger, pylogging.LogLevel.INFO)
        try:
            test_V_t.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder {}? (yes / no)".format(test_V_t.folder))
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(test_V_t.folder)
            raise e
    
    if parameters["run_V_reset"]:
        test_V_reset = lif.Test_V_reset(neurons, sthal, parameters)
        pylogging.set_loglevel(test_V_reset.logger, pylogging.LogLevel.INFO)
        try:
            test_V_reset.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder {}? (yes / no)".format(test_V_reset.folder))
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(test_V_reset.folder)
            raise e
    
    
    if parameters["run_I_gl"]:
        test_I_gl = lif.Test_g_L(neurons, sthal, parameters)
        pylogging.set_loglevel(test_I_gl.logger, pylogging.LogLevel.INFO)
        #try:
        test_I_gl.run_experiment()
        #except Exception,e:
        #    print "ERROR: ", e
        #    delete = raw_input("Delete folder {}? (yes / no)".format(test_I_gl.folder))
        #    if delete in ("yes","Yes","y","Y"):
        #        shutil.rmtree(test_I_gl.folder)
        #    raise e




quit()

