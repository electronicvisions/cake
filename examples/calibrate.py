"""Runs E_l calibration, plots and saves data"""

import pyhalbe
import pycairo.experiment
from pycairo.experiment import Voltage, Current
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # execute before using pyplot without X
import matplotlib.pyplot as plt

from pycairo.helpers.calibtic import init_backend as init_calibtic
from pycairo.helpers.redman import init_backend as init_redman
from pycairo.helpers.sthal import StHALContainer

import pyoneer  # needs to be imported for get_pyoneer()

import pickle

import pylogging

# Disable most log messages
pylogging.default_config(pylogging.LogLevel.ERROR)

# config
neurons = range(512)

# Calibtic and Redman backends
backend = init_calibtic(path='/afsuser/weilbach/calibtic_config/')
backend_r = init_redman(path='/afsuser/weilbach/redman_config/')

# StHAL
sthal = StHALContainer()

# E_l calibration
class Calibrate_E_l_zero_current(pycairo.experiment.Calibrate_E_l):
    """E_l calibration with zero currents."""
    def get_parameters(self):
        parameters = super(Calibrate_E_l_zero_current, self).get_parameters()
        for neuron_id in self.get_neurons():
            parameters[neuron_id].update({
                pyhalbe.HICANN.neuron_parameter.I_gl: Current(1000),
                pyhalbe.HICANN.neuron_parameter.V_t: Voltage(1700),
                pyhalbe.HICANN.neuron_parameter.I_convi: Current(0),
                pyhalbe.HICANN.neuron_parameter.I_convx: Current(0)
            })
        return parameters

    def init_experiment(self):
        super(Calibrate_E_l_zero_current, self).init_experiment()
        self.repetitions = 3

    def get_steps(self):
        steps = []
        for voltage in [300, 400, 500, 600, 700, 800, 900]:
            steps.append({pyhalbe.HICANN.neuron_parameter.E_l: Voltage(voltage)})
        return defaultdict(lambda: steps)



#e = Calibrate_E_l_zero_current(neurons, sthal_container=sthal, calibtic_backend=backend, redman_backend=backend_r)
#e.run_experiment()

# plot and save
#save_data = {}

#for neuron_id in e.get_neurons():
#    steps = [step[pyhalbe.HICANN.neuron_parameter.E_l].value for step in e.get_steps()[neuron_id]]
#    plt.errorbar(e.results_mean[neuron_id], steps, xerr=e.results_std[neuron_id], fmt='.')
#    x = np.linspace(min(e.results_mean[neuron_id])-50, max(e.results_mean[neuron_id])+50)
#    y = map(e.results_polynomial[neuron_id].apply, x)
#    save_data["meas_x{}".format(neuron_id)] = e.results_mean[neuron_id]
#    save_data["meas_xerr{}".format(neuron_id)] = e.results_std[neuron_id]
#    save_data["meas_y{}".format(neuron_id)] = steps
#    save_data["fit_x{}".format(neuron_id)] = x
#    save_data["fit_y{}".format(neuron_id)] = y
#    plt.plot(x, y)
#plt.savefig('calib_E_l_zero_current.png')
#np.savez_compressed("calib_E_l_zero_current.npz", **save_data)

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

## V_reset calibration
e = pycairo.experiment.Calibrate_V_reset(
        neurons,
        sthal_container=sthal,
        calibtic_backend=backend,
        redman_backend=backend_r,
        loglevel=pylogging.LogLevel.INFO)
e.run_experiment()

## plot and save
#save_data = {}

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
# np.savez_compressed("calib_V_reset.npz", **save_data)
pickle.dump(e.all_results, open("calib_V_reset.raw", 'wb'))
