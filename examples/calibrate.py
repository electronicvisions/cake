"""Runs E_l calibration, plots and saves data"""

import pyhalbe
import pycairo.experiment

import numpy as np
import matplotlib
matplotlib.use('Agg')  # execute before using pyplot without X
import matplotlib.pyplot as plt

from pycairo.helpers.calibtic import init_backend as init_calibtic
from pycairo.helpers.redman import init_backend as init_redman
from pycairo.helpers.sthal import StHALContainer

import pyoneer  # needs to be imported for get_pyoneer()

# disable Scheriff because there are currently
# many false positive messages
pneer = pyhalbe.Handle.get_pyoneer()
pneer.useScheriff = False

# config
neurons = [0,1,2,3] # range(512)

# Calibtic and Redman backends
backend = init_calibtic(path='/afsuser/weilbach/calibtic_config/')
backend_r = init_redman(path='/afsuser/weilbach/redman_config/')

# StHAL
sthal = StHALContainer()

# E_l calibration
e = pycairo.experiment.Calibrate_g_L(neurons, sthal_container=sthal, calibtic_backend=backend, redman_backend=backend_r)
e.run_experiment()

# plot and save
save_data = {}

for neuron_id in e.get_neurons():
    steps = [step[pyhalbe.HICANN.neuron_parameter.I_gl].value for step in e.get_steps()[neuron_id]]
    plt.errorbar(e.results_mean[neuron_id], steps, xerr=e.results_std[neuron_id], fmt='.')
    x = np.linspace(min(e.results_mean[neuron_id])-50, max(e.results_mean[neuron_id])+50)
    y = map(e.results_polynomial[neuron_id].apply, x)
    save_data["meas_x{}".format(neuron_id)] = e.results_mean[neuron_id]
    save_data["meas_xerr{}".format(neuron_id)] = e.results_std[neuron_id]
    save_data["meas_y{}".format(neuron_id)] = steps
    save_data["fit_x{}".format(neuron_id)] = x
    save_data["fit_y{}".format(neuron_id)] = y
    plt.plot(x, y)
plt.savefig('calib_g_l.png')
np.savez_compressed("calib_g_l.npz", **save_data)

## V_t calibration
#e = pycairo.experiment.Calibrate_V_t(neurons, sthal_container=sthal, calibtic_backend=backend)
#e.run_experiment()
#
## plot and save
#save_data = {}
#
#for neuron_id in e.get_neurons():
#    steps = [step[pyhalbe.HICANN.neuron_parameter.V_t].value for step in e.get_steps()[neuron_id]]
#    plt.errorbar(e.results_mean[neuron_id], steps, xerr=e.results_std[neuron_id], fmt='.')
#    x = np.linspace(min(e.results_mean[neuron_id])-50, max(e.results_mean[neuron_id])+50)
#    y = map(e.results_polynomial[neuron_id].apply, x)
#    save_data["meas_x{}".format(neuron_id)] = e.results_mean[neuron_id]
#    save_data["meas_xerr{}".format(neuron_id)] = e.results_std[neuron_id]
#    save_data["meas_y{}".format(neuron_id)] = steps
#    save_data["fit_x{}".format(neuron_id)] = x
#    save_data["fit_y{}".format(neuron_id)] = y
#    plt.plot(x, y, label='2nd order polynomial fit')
#plt.savefig('calib_V_t.png')
#np.savez_compressed("calib_V_t.npz", **save_data)
#
## V_reset calibration
#e = pycairo.experiment.Calibrate_V_reset(neurons, sthal_container=sthal, calibtic_backend=backend)
#e.run_experiment()
#
## plot and save
#save_data = {}
#
#for neuron_id in e.get_neurons():
#    steps = [step[pyhalbe.HICANN.shared_parameter.V_reset].value for step in e.get_steps()[neuron_id]]
#    plt.errorbar(e.results_mean[neuron_id], steps, xerr=e.results_std[neuron_id], fmt='.')
#    x = np.linspace(min(e.results_mean[neuron_id])-50, max(e.results_mean[neuron_id])+50)
#    y = map(e.results_polynomial[neuron_id].apply, x)
#    save_data["meas_x{}".format(neuron_id)] = e.results_mean[neuron_id]
#    save_data["meas_xerr{}".format(neuron_id)] = e.results_std[neuron_id]
#    save_data["meas_y{}".format(neuron_id)] = steps
#    save_data["fit_x{}".format(neuron_id)] = x
#    save_data["fit_y{}".format(neuron_id)] = y
#    plt.plot(x, y, label='2nd order polynomial fit')
#plt.savefig('calib_V_reset.png')
#np.savez_compressed("calib_V_reset.npz", **save_data)
