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
parser.add_argument('--calibpath', type=str, default="/wang/data/calibration/")
parser.add_argument('--anaonly', action="store_true", default=False)
parser.add_argument('--extrasuffix', type=str, default=None)
parser.add_argument('--correctoffset', action="store_true", default=False)
args = parser.parse_args()

WAFER = args.wafer
HICANN = args.hicann
FREQ = args.freq
BKGISI = args.bkgisi

suffix='_'.join(["w{}","h{}","f{}","bkgisi{}"]).format(WAFER,HICANN,FREQ,BKGISI)
if args.extrasuffix:
    suffix += "_"+args.extrasuffix

PATH = '_'.join(["defects", suffix])
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

defects_file = open(os.path.join(PATH, '_'.join(["defects",suffix])+'.csv'), 'a')
defects = csv.writer(defects_file, dialect='excel-tab')

def ana(spikes):

    print "analyzing data of driver", driver

    spikes = np.loadtxt(os.path.join(PATH, '_'.join(['spikes',"drv",str(driver),suffix])+".dat"))

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

    details_file = open(os.path.join(PATH, '_'.join(['details',"drv",str(driver),suffix])+'.csv'), 'w')
    details = csv.writer(details_file, dialect='excel-tab')

    offset=sorted(spikes[:,0])[0] if args.correctoffset else 0

    for i in range(64):
        plt.axhline(i, c='0.8', ls=':')
        try:
            s = spikes[spikes[:,1] == i]

            pos = 5e-6 + i*50e-6

            try:
                correct = s[np.abs(s[:,0] - pos - 25e-6 - offset) <= 25e-6 * 1.1,:]
            except:
                correct = np.array(())
            try:
                incorrect = s[np.abs(s[:,0] - pos - 25e-6  - offset) > 25e-6 * 1.1,:]
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

    plt.savefig(os.path.join(PATH, '_'.join(["defects","drv",str(driver),suffix])+".pdf"))

    print "analyzing done"

def aquire(driver):

    print "aquiring data for driver", driver

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
        #default
        half = shallow.BOT if addr < 32 else shallow.TOP

        #alternative
        #half = shallow.TOP if addr < 32 else shallow.BOT
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

    np.savetxt(os.path.join(PATH, '_'.join(['spikes',"drv",str(driver),suffix])+".dat"), spikes)

    print "aquiring done"

if __name__ == "__main__":

    run_on_hardware = not args.anaonly

    if run_on_hardware:

        hardware = shallow.Hardware(WAFER, HICANN,
                                    os.path.join(args.calibpath,'wafer_{0}').format(WAFER), FREQ*1e6, BKGISI)
        hardware.connect()

        neuron_blacklist = np.loadtxt('blacklist_w{}_h{}.csv'.format(WAFER, HICANN)) # no notion of freq and bkgisi

        # Configure floating gates and synapse drivers
        hardware.set_parameters(params)

    for driver in range(0, 224):

        if run_on_hardware:
            aquire(driver)
        ana(driver)
