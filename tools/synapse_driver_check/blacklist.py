#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shallow


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('hicann', type=int)
parser.add_argument('--wafer', type=int, default=0)
args = parser.parse_args()

WAFER = args.wafer
HICANN = args.hicann

#PATH = 'blacklists/{0}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
#os.mkdir(PATH)

from pycake.helpers.units import DAC, Voltage, Current

params = shallow.Parameters()

for i in range(512):

    params.neuron[i].E_l = Voltage(700).toDAC().value
    params.neuron[i].V_t = Voltage(730).toDAC().value
    params.neuron[i].E_synx = Voltage(800).toDAC().value
    params.neuron[i].E_syni = Voltage(600).toDAC().value
    params.neuron[i].V_syntcx = 800
    params.neuron[i].V_syntci = 800
    params.neuron[i].I_gl = 409

params.shared.V_reset = 200

hardware = shallow.Hardware(WAFER, HICANN,
        '/wang/data/calibration/wafer_{0}'.format(WAFER))
hardware.connect()

blacklist_file = open('blacklist_w{}_h{}.csv'.format(WAFER,HICANN), 'a')
blacklist = csv.writer(blacklist_file, dialect='excel-tab')

# Configure floating gates and synapse drivers
hardware.set_parameters(params)

for neurons in [range(i, i+32) for i in range(0, 512, 32)]:
    hardware.clear_routes()
    hardware.clear_spike_trains()
    
    # Choose a driver and calculate the corresponding bus.
    if neurons[0] < 256:
        receiving_link = (neurons[0] / 32)
        bus = (receiving_link + 1) % 8
        driver = bus * 2
    else:
        receiving_link = (neurons[0] % 256 / 32)
        bus = (receiving_link + 1) % 8
        driver = 127 - bus * 2
    assert bus == shallow.get_bus_from_driver(driver)

    sending_addr = 1

    for addr, neuron in enumerate(neurons):
        hardware.add_route(
            sending_addr,
            driver,
            neuron
            )
        hardware.assign_address(neuron, addr + 1)
    hardware.enable_readout(receiving_link)

    # Create input spike trains
    train = np.arange(200) * 0.1e-6 + 5e-6
    hardware.add_spike_train(bus, sending_addr, train)

    record_membrane = False

    if record_membrane:
        adc = 0
        hardware.enable_adc(neurons[0], adc)
        timestamps, trace = hardware.record(adc, 1e-3)
        np.savetxt('membrane_n_{}_w{}_h{}.txt'.format(neurons[0], WAFER, HICANN), np.transpose([timestamps, trace]))
    else:
        hardware.run(1e-3)

    spikes = hardware.get_spikes(receiving_link)

    for i, nrn in enumerate(neurons):
        correct = spikes[(spikes[:,1] == i+1) & (spikes[:,0] <= 30e-6),:]
        incorrect = spikes[(spikes[:,1] == i+1) & (spikes[:,0] > 30e-6),:]

        n_correct = len(correct)
        n_incorrect = len(incorrect)

        if n_correct != 0 and n_incorrect == 0:
            print "Neuron {0} works fine.".format(nrn), "n_correct:", n_correct, "n_incorrect:", n_incorrect
        else:
            print "Neuron {0} is dead.".format(nrn), "n_correct:", n_correct, "n_incorrect:", n_incorrect
            blacklist.writerow((nrn, ))
            blacklist_file.flush()
