#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pylogging
pylogging.default_config(date_format='absolute')
pylogging.set_loglevel(pylogging.get("sthal"), pylogging.LogLevel.INFO)
pylogging.set_loglevel(pylogging.get("sthal.HICANNConfigurator.Time"), pylogging.LogLevel.DEBUG)
pylogging.set_loglevel(pylogging.get("Default"), pylogging.LogLevel.INFO)

import shallow

from pycake.helpers.misc import mkdir_p

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('hicann', type=int)
parser.add_argument('--wafer', type=int, default=0)
parser.add_argument('--freq', type=int, default=100)
parser.add_argument('--bkgisi', type=int, default=10000)
args = parser.parse_args()

WAFER = args.wafer
HICANN = args.hicann
FREQ = args.freq
BKGISI = args.bkgisi

PATH = 'defects_w{}_h{}_f{}_bkgisi{}'.format(WAFER,HICANN,FREQ,BKGISI)
mkdir_p(PATH)

from pycake.helpers.units import DAC, Voltage, Current

params = shallow.Parameters()

params.neuron.E_l = Voltage(700).toDAC().value
params.neuron.V_t = Voltage(730).toDAC().value
params.neuron.E_synx = Voltage(800).toDAC().value
params.neuron.E_syni = Voltage(600).toDAC().value
params.neuron.V_syntcx = 800
params.neuron.V_syntci = 800
params.neuron.I_gl = 409


params.shared.V_reset = 200

hardware = shallow.Hardware(WAFER, HICANN,
                            '/wang/data/calibration/wafer_{0}'.format(WAFER), FREQ*1e6, BKGISI)
hardware.connect()

neuron_blacklist = np.loadtxt('blacklist_w{}_h{}.csv'.format(WAFER, HICANN))

# Configure floating gates and synapse drivers
hardware.set_parameters(params)

defects_file = open(os.path.join(PATH, 'defects_w{}_h{}.csv'.format(WAFER, HICANN)), 'a')
defects = csv.writer(defects_file, dialect='excel-tab')

for driver in range(0, 224):
    hardware.clear_routes()
    hardware.clear_spike_trains()
    
    # Every synapse driver is accessible via a specific bus. Calculate that bus.
    bus = shallow.get_bus_from_driver(driver)
    
    if driver < 112:
        neurons = np.arange(256)
    else:
        neurons = np.arange(256, 512)

    neurons = np.delete(neurons, neuron_blacklist)
    for i in np.arange(32) + bus*32:
        neurons = neurons[neurons != i]
    for i in np.arange(32) + bus*32 + 256:
        neurons = neurons[neurons != i]

    even = neurons[neurons % 2 == 0]
    odd = neurons[neurons % 2 == 1]

    recording_links = []
    for addr in range(64):
        half = shallow.BOT if addr < 32 else shallow.TOP
        if addr % 32 < 16:
            neuron = even[addr]
        else:
            neuron = odd[addr]
        hardware.add_route(
            int(addr),
            driver,
            int(neuron),
            half
            )
        hardware.assign_address(int(neuron), int(addr))

        rl = int((neuron % 256) / 32)
        if rl not in recording_links:
            recording_links.append(rl)

    for rl in recording_links:
        hardware.enable_readout(rl)

    # Create input spike trains
    for i in range(64):
        train = np.arange(200) * 0.1e-6 + 5e-6 + 50e-6*i
        hardware.add_spike_train(bus, i, train)
            
    hardware.run(4e-3)
    spikes = np.vstack([hardware.get_spikes(rl) for rl in recording_links])
    
    plt.figure()
    plt.title("Test Result for Synapse Driver {0}".format(driver))
    plt.xlabel("Time [s]")
    plt.ylabel("Source Address")

    plt.grid(False)
    plt.ylim((-1, 64))
    plt.xlim((0, 4e-3))
    plt.yticks(range(0, 64, 8))

    result = []

    defect_addresses = 0

    details_file = open(os.path.join(PATH, 'details_{0}.csv'.format(driver)), 'a')
    details = csv.writer(details_file, dialect='excel-tab')

    for i in range(64):
        plt.axhline(i, c='0.8', ls=':')
        try:
            s = spikes[spikes[:,1] == i]
            pos = 5e-6 + i*50e-6
            
            try:
                correct = s[np.abs(s[:,0] - pos - 25e-6) <= 25e-6,:]
            except:
                correct = np.array(())
            try:
                incorrect = s[np.abs(s[:,0] - pos - 25e-6) > 25e-6,:]
            except:
                incorrect = np.array(())

            details.writerow((i, correct.size/2, incorrect.size/2))
            details_file.flush()

            if correct.size == 0 or incorrect.size != 0:
                defect_addresses += 1

            plt.plot(correct[:,0], correct[:,1], 'g|')
            plt.plot(incorrect[:,0], incorrect[:,1], 'r|')
        except BaseException, e:
            pass

    defects.writerow((driver, defect_addresses))
    defects_file.flush()

    plt.savefig(os.path.join(PATH, 'defects_{0}.pdf'.format(driver)))
