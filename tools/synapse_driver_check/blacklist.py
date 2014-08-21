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

WAFER = 0
HICANN = 84

#PATH = 'blacklists/{0}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
#os.mkdir(PATH)

params = shallow.Parameters()
params.neuron.E_l = 400
params.neuron.V_t = 500
params.neuron.E_synx = 1023
params.neuron.E_syni = 330
params.neuron.V_syntcx = 800
params.neuron.V_syntci = 800
params.neuron.I_gl = 200

params.shared.V_reset = 200

hardware = shallow.Hardware(WAFER, HICANN,
        '/wang/data/calibration/wafer_{0}'.format(WAFER))
hardware.connect()

blacklist_file = open('blacklist.csv', 'a')
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

    for addr, neuron in enumerate(neurons):
        hardware.add_route(
            42,
            driver,
            neuron
            )
        hardware.assign_address(neuron, addr + 1)
    hardware.enable_readout(receiving_link)

    # Create input spike trains
    train = np.arange(200) * 0.1e-6 + 5e-6
    hardware.add_spike_train(bus, 42, train)
            
    hardware.run(1e-3)
    spikes = hardware.get_spikes(receiving_link)

    for i, nrn in enumerate(neurons):
        correct = spikes[(spikes[:,1] == i+1) & (spikes[:,0] <= 30e-6),:]
        incorrect = spikes[(spikes[:,1] == i+1) & (spikes[:,0] > 30e-6),:]
        if correct.size != 0 and incorrect.size == 0:
            print "Neuron {0} works fine.".format(nrn)
        else:
            print "Neuron {0} is dead.".format(nrn)
            blacklist.writerow((nrn, ))
            blacklist_file.flush()
