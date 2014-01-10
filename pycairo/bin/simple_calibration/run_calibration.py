"""Runs E_l calibration, plots and saves data"""

from pycairo.helpers.calibtic import init_backend as init_calibtic
from pycairo.helpers.redman import init_backend as init_redman
from pycairo.helpers.sthal import StHALContainer

import cairotest_experiments as experiments
from parameters import parameters
import shutil

import pylogging

import sys

# config
neurons = range(512)

# Calibtic and Redman backends
backend_c = init_calibtic(path=parameters["backend_c"])
backend_r = init_redman(path=parameters["backend_r"])

coord_wafer  = parameters["coord_wafer"]
coord_hicann = parameters["coord_hicann"]

# StHAL
sthal = StHALContainer(coord_wafer, coord_hicann)

default_logger = pylogging.get("Default")
pylogging.set_loglevel(default_logger, pylogging.LogLevel.ERROR)


if parameters["calibrate"]:
    if parameters["run_E_l"]:
        calib_E_l = experiments.Calibrate_E_l(neurons, sthal, backend_c, backend_r)
        pylogging.set_loglevel(calib_E_l.logger, pylogging.LogLevel.INFO)
        try:
            calib_E_l.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder? (yes/no): ")
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(calib_E_l.folder)
            raise e
    
    if parameters["run_V_t"]:
        calib_V_t = experiments.Calibrate_V_t(neurons, sthal, backend_c, backend_r)
        pylogging.set_loglevel(calib_V_t.logger, pylogging.LogLevel.INFO)
        try:
            calib_V_t.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder? (yes/no): ")
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(calib_V_t.folder)
            raise e
    
    if parameters["run_V_reset"]:
        calib_V_reset = experiments.Calibrate_V_reset(neurons, sthal, backend_c, backend_r)
        pylogging.set_loglevel(calib_V_reset.logger, pylogging.LogLevel.INFO)
        try:
            calib_V_reset.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder? (yes/no): ")
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(calib_V_reset.folder)
            raise e
    
    if parameters["run_I_gl"]:
        calib_I_gl = experiments.Calibrate_g_L(neurons, sthal, backend_c, backend_r)
        pylogging.set_loglevel(calib_I_gl.logger, pylogging.LogLevel.INFO)
        #try:
        calib_I_gl.run_experiment()
        #except:
        #    print "ERROR: ", e
        #    delete = raw_input("Delete folder? (yes/no): ")
        #    if delete in ("yes","Yes","y","Y"):
        #        shutil.rmtree(calib_I_gl.folder)
        #    raise e

    if parameters["run_E_synx"]:
        calib_E_synx = experiments.Calibrate_E_synx(neurons, sthal, backend_c, backend_r)
        pylogging.set_loglevel(calib_E_synx.logger, pylogging.LogLevel.INFO)
        try:
            calib_E_synx.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder? (yes/no): ")
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(calib_E_synx.folder)
            raise e

    if parameters["run_E_syni"]:
        calib_E_syni = experiments.Calibrate_E_syni(neurons, sthal, backend_c, backend_r)
        pylogging.set_loglevel(calib_E_syni.logger, pylogging.LogLevel.INFO)
        try:
            calib_E_syni.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder? (yes/no): ")
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(calib_E_syni.folder)
            raise e

if parameters["measure"]:
    if parameters["run_E_l"]:
        test_E_l = experiments.Test_E_l(neurons, sthal, backend_c, backend_r)
        pylogging.set_loglevel(test_E_l.logger, pylogging.LogLevel.INFO)
        try:
            test_E_l.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder? (yes/no): ")
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(test_E_l.folder)
            raise e
    
    if parameters["run_V_t"]:
        test_V_t = experiments.Test_V_t(neurons, sthal, backend_c, backend_r)
        pylogging.set_loglevel(test_V_t.logger, pylogging.LogLevel.INFO)
        try:
            test_V_t.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder? (yes/no): ")
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(test_V_t.folder)
            raise e
    
    if parameters["run_V_reset"]:
        test_V_reset = experiments.Test_V_reset(neurons, sthal, backend_c, backend_r)
        pylogging.set_loglevel(test_V_reset.logger, pylogging.LogLevel.INFO)
        try:
            test_V_reset.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder? (yes/no): ")
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(test_V_reset.folder)
            raise e
    
    
    if parameters["run_I_gl"]:
        test_I_gl = experiments.Test_g_L(neurons, sthal, backend_c, backend_r)
        pylogging.set_loglevel(test_I_gl.logger, pylogging.LogLevel.INFO)
        #try:
        test_I_gl.run_experiment()
        #except Exception,e:
        #    print "ERROR: ", e
        #    delete = raw_input("Delete folder? (yes/no): ")
        #    if delete in ("yes","Yes","y","Y"):
        #        shutil.rmtree(test_I_gl.folder)
        #    raise e

    if parameters["run_E_synx"]:
        test_E_synx = experiments.Test_E_synx(neurons, sthal, backend_c, backend_r)
        pylogging.set_loglevel(calib_E_synx.logger, pylogging.LogLevel.INFO)
        try:
            test_E_synx.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder? (yes/no): ")
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(test_E_synx.folder)
            raise e

    if parameters["run_E_syni"]:
        test_E_syni = experiments.Test_E_syni(neurons, sthal, backend_c, backend_r)
        pylogging.set_loglevel(test_E_syni.logger, pylogging.LogLevel.INFO)
        try:
            test_E_syni.run_experiment()
        except Exception,e:
            print "ERROR: ", e
            delete = raw_input("Delete folder? (yes/no): ")
            if delete in ("yes","Yes","y","Y"):
                shutil.rmtree(test_E_syni.folder)
            raise e



quit()

