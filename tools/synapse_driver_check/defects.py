#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import csv
import json

from collections import defaultdict

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

import glob

import random

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

for i in range(512):

    params.neuron[i].E_l = Voltage(700).toDAC().value
    params.neuron[i].V_t = Voltage(740).toDAC().value
    params.neuron[i].E_synx = Voltage(800).toDAC().value
    params.neuron[i].E_syni = Voltage(600).toDAC().value
    params.neuron[i].V_syntcx = 800
    params.neuron[i].V_syntci = 800
    params.neuron[i].I_gl = 409

params.shared.V_reset = 200

defects_file = open(os.path.join(PATH, '_'.join(["defects",suffix])+'.csv'), 'w')
defects = csv.writer(defects_file, dialect='excel-tab')

import resource

def ana(driver, filename_stub):

    print "analyzing data of driver", driver

    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if not filename_stub:
        guess = glob.glob(os.path.join(PATH,"spikes*drv_{}_*.dat".format(driver)))[0]
        guess = guess.split('/')[-1]
        guess = guess.lstrip("spikes_")
        guess = guess.rstrip(".dat")

        filename_stub = guess

    with open(os.path.join(PATH,'addr_neuron_map_'+filename_stub+'.csv')) as f:
        reader = csv.DictReader(f, quoting=csv.QUOTE_NONNUMERIC)

        for row in reader:
            addr_neuron_map = row

    spikes = np.loadtxt(os.path.join(PATH, "spikes_"+filename_stub+".dat"))

    fig, ax1 = plt.subplots()
    plt.title("Test Result for Synapse Driver {0}".format(driver))
    plt.xlabel("Time [s]")
    plt.ylabel("Source Address (Neuron, OB)")

    plt.grid(False)
    plt.ylim((-1, 64))
    plt.xlim((0, 4e-3))
    plt.yticks(range(0, 64, 1))

    yticklabels=["{:.0f} ({:.0f}, {})".format(yt, addr_neuron_map[yt], int((addr_neuron_map[yt] % 256) / 32)) for yt in ax1.get_yticks()]

    ax1.set_yticklabels(yticklabels)

    plt.tick_params(axis='y', which='both', labelsize=5)

    result = []

    defect_addresses = 0

    details_file = open(os.path.join(PATH, "detail_"+filename_stub+".csv"), 'w')
    details = csv.writer(details_file, dialect='excel-tab')

    if args.correctoffset:
        try:
            offset=sorted(spikes[spikes[:,1] == 1][:,0])[0]
            offset -= 3*25e-6
        except IndexError:
            try:
                offset=sorted(spikes[spikes[:,1] == 2][:,0])[0]
                offset -= 1*25e-6
            except IndexError:
                offset = 0
    else:
            offset = 0
    print "offset", offset

    right_yticklabels = []

    for i in range(64):
        plt.axhline(i, c='0.8', ls=':')
        try:
            s = spikes[spikes[:,1] == i]

            pos = 5e-6 + i*50e-6

            safety=3

            try:
                correct = s[np.abs(s[:,0] - pos - 25e-6*safety - offset) <= 25e-6*safety,:]
            except:
                correct = np.array(())
            try:
                incorrect = s[np.abs(s[:,0] - pos - 25e-6*safety  - offset) > 25e-6*safety,:]
            except:
                incorrect = np.array(())

            right_yticklabels.append("{},{}".format(correct.size/2,incorrect.size/2))

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
    ax2 = ax1.twinx()
    ax2.set_ylabel("cor,incor")
    plt.ylim((-1, 64))
    plt.yticks(range(0, 64, 1))
    ax2.set_yticklabels(right_yticklabels)
    plt.tick_params(axis='y', which='both', labelsize=5)

    plt.savefig(os.path.join(PATH, "defects_"+filename_stub+".pdf"))
    
    print os.path.join(PATH, "defects_"+filename_stub+".pdf")

    plt.close()

    print "analyzing done"

def aquire(driver):

def get_neurons(driver):

    if driver < 112:
        neurons = np.arange(256)
    else:
        neurons = np.arange(256, 512)

    bus = shallow.get_bus_from_driver(driver)

    neurons = np.delete(neurons, neuron_blacklist)

    random.seed(driver)
    random.shuffle(neurons)

    for i in np.arange(32) + bus*32:
        neurons = neurons[neurons != i]
    for i in np.arange(32) + bus*32 + 256:
        neurons = neurons[neurons != i]

    even = list(neurons[neurons % 2 == 0])
    odd = list(neurons[neurons % 2 == 1])

    return even, odd

def get_half_and_bank(addr, even, odd):

    msb = (addr & 0x30) >> 4

    foo = {0 : (shallow.BOT, even),
           1 : (shallow.TOP, odd),
           2 : (shallow.BOT, odd),
           3 : (shallow.TOP, even)}

    half, bank = foo[msb]

    return half, bank

def prepare_drivers(drivers):

    print "preparing drivers"

    for driver in drivers:

        even, odd = get_neurons(driver)

        for addr in range(64):

            half, bank = get_half_and_bank(addr, even, odd)

            neuron = bank[addr]
            max_index = addr

            hardware.setup_driver(int(addr),
                                  driver,
                                  int(neuron),
                                  half)

        #print hardware.hicann

    print "done"

def aquire(driver):

    print "aquiring data for driver", driver

    hardware.clear_routes()
    hardware.clear_spike_trains()

    # Every synapse driver is accessible via a specific bus. Calculate that bus.
    bus = shallow.get_bus_from_driver(driver)

    even, odd = get_neurons(driver)

    recording_links = []

    max_index = 0

    addr_neuron_map = {}

    for addr in range(64):

        half, bank = get_half_and_bank(addr, even, odd)
        #print msb, half, len(bank)
        neuron = bank[addr]
        max_index = addr

        bus, hline, vline =  hardware.add_route(int(addr),
                                                 driver,
                                                 int(neuron),
                                                 half)
        print "bus {}, hline {}, vline {}".format(bus, hline, vline)

        hardware.assign_address(int(neuron), int(addr))

        addr_neuron_map[addr] = int(neuron)

        rl = int((neuron % 256) / 32)
        if rl not in recording_links:
            recording_links.append(rl)

    for rl in recording_links:
        hardware.enable_readout(rl)

    # Create input spike trains
    for i in range(max_index+1):
        train = np.arange(200) * 0.1e-6 + 5e-6 + 50e-6*i
        hardware.add_spike_train(bus, i, train)

    print hardware.hicann

    duration = 4e-3

    record_membrane = False

    filename_stub = '_'.join(["bus",str(bus),"hline", str(hline.value()), "vline", str(vline.value()), "drv",str(driver), suffix])

    #record_addr = 56

    #params.neuron[addr_neuron_map[record_addr]].V_t = 1023

    # Configure floating gates and synapse drivers
    hardware.set_parameters(params)

    if record_membrane:
        adc = 0
        hardware.enable_adc(addr_neuron_map[record_addr], adc)
        timestamps, trace = hardware.record(adc, duration)
        np.savetxt(os.path.join(PATH, 'membrane_n{}_'.format(addr_neuron_map[record_addr])+filename_stub+'.dat'), np.transpose([timestamps, trace]))
    else:
        hardware.run(duration)

    rl_dict = {}

    for n, rl in enumerate(recording_links):
        spikes = np.array(hardware.get_spikes(rl))
        rl_dict[rl] = spikes[:,1]

    addr_dict = defaultdict(set)

    for rl, rl_spikes in rl_dict.iteritems():

        for addr in rl_spikes:
            addr_dict[addr].add(rl)

    for addr, rls in addr_dict.iteritems():
        print "addr {}, rls {}, len(rls) {}".format(addr, rls, len(rls))
        if len(rls) > 1 and addr != 0:
            #raise Exception("WTF")
            print "PROBLEM(?)"

    spikes = np.vstack([hardware.get_spikes(rl) for rl in recording_links])

    with open(os.path.join(PATH,'addr_neuron_map_'+filename_stub+'.csv'), 'wb') as f:
        w = csv.DictWriter(f, addr_neuron_map.keys())
        w.writeheader()
        w.writerow(addr_neuron_map)

    np.savetxt(os.path.join(PATH, "spikes_"+filename_stub+".dat"), spikes)

    print hardware.hicann
    print hardware.hicann.layer1

    hardware.wafer.dump(os.path.join(PATH, "wafer_"+filename_stub+".xml"), True)

    print "aquiring done"

    return filename_stub

def store_voltages(filename):

    cmd = "ssh -F /dev/null -x -i ./id_rsa_resetuser resetuser@raspeval-001 -o PubkeyAuthentication=yes /home/pi/voltages/readVoltages"
    data = dict()
    for l in os.popen(cmd).read().rstrip('\n').split('\n'):
        if "dac" in l: break
        if "voltage" in l: continue
        data.update({ l.split('\t')[0]: map(float, l.split('\t')[1:]) })

    with open(filename, "w") as f:

        json.dump(data, f)

if __name__ == "__main__":

    random.seed(1)

    run_on_hardware = not args.anaonly

    neuron_blacklist = np.loadtxt('blacklist_w{}_h{}.csv'.format(WAFER, HICANN)) # no notion of freq and bkgisi

    hardware = shallow.Hardware(WAFER, HICANN, os.path.join(args.calibpath,'wafer_{0}').format(WAFER), FREQ*1e6, BKGISI)

    if run_on_hardware:

        hardware.connect()


    drivers = range(224)

    if run_on_hardware:
        prepare_drivers(drivers)

    for driver in drivers:

        filename_stub = None

        if run_on_hardware:
            filename_stub = aquire(driver)
            store_voltages(os.path.join(PATH, "voltages_"+filename_stub+".json"))
