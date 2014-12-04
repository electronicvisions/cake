#! /usr/bin/python
# -*- coding: utf-8 -*-

from neo.core import (Block, Segment, RecordingChannelGroup,
                      RecordingChannel, AnalogSignal, SpikeTrain)

from neo.io import NeoHdf5IO

import quantities

import os
import time

from collections import defaultdict

import numpy as np

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

import ana_defects

parser = argparse.ArgumentParser()
parser.add_argument('hicann', type=int)
parser.add_argument('--wafer', type=int, default=0)
parser.add_argument('--freq', type=int, default=100)
parser.add_argument('--bkgisi', type=int, default=10000)
parser.add_argument('--calibpath', type=str, default="/wang/data/calibration/")
parser.add_argument('--extrasuffix', type=str, default=None)
parser.add_argument('--dumpwafercfg', action="store_true", default=False)
parser.add_argument('--ana', action="store_true", default=False)
parser.add_argument('--drivers', type=int, nargs="+", default=range(224))
parser.add_argument('--ninputspikes', type=int, default=200)
parser.add_argument('--inputspikeisi', type=float, default=0.1e-6)
parser.add_argument('--addroffset', type=float, default=50e-6)
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

import resource

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

def aquire(seg, driver):

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
        #print "bus {}, hline {}, vline {}".format(bus, hline, vline)

        hardware.assign_address(int(neuron), int(addr))

        addr_neuron_map[addr] = int(neuron)

        rl = int((neuron % 256) / 32)
        if rl not in recording_links:
            recording_links.append(rl)

    for rl in recording_links:
        hardware.enable_readout(rl)

    # Create input spike trains

    start_offset = 5e-6
    addr_offset = args.addroffset
    n_input_spikes = args.ninputspikes
    input_spike_isi = args.inputspikeisi

    seg.annotations["start_offset"] = start_offset
    seg.annotations["addr_offset"] = addr_offset
    seg.annotations["addr_neuron_map"] = addr_neuron_map
    seg.annotations["n_input_spikes"] = n_input_spikes
    seg.annotations["input_spike_isi"] = input_spike_isi

    duration = 4e-3

    for i in range(max_index+1):
        train = np.arange(n_input_spikes) * input_spike_isi + start_offset + addr_offset*i
        duration=max(train)+0.001
        hardware.add_spike_train(bus, i, train)

    #print hardware.hicann
    seg.annotations["duration"] = duration


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
        #print "addr {}, rls {}, len(rls) {}".format(addr, rls, len(rls))
        if len(rls) > 1 and addr != 0:
            #raise Exception("WTF")
            print "PROBLEM(?)"

    spikes = np.vstack([hardware.get_spikes(rl) for rl in recording_links])

    for i in range(64):

        try:
            spikes_i = spikes[spikes[:,1] == i][:,0]
        except IndexError:
            spikes_i = []

        if len(spikes_i) != 0: # any spikes?
            seg.spiketrains.append(SpikeTrain(spikes_i*quantities.s, t_stop=spikes_i[-1]+1, addr=i))
        else:
            seg.spiketrains.append(SpikeTrain([]*quantities.s, t_stop=[], addr=i)) # empty spike train

        #print i, seg.spiketrains[-1].times

    if args.dumpwafercfg:
        hardware.wafer.dump(os.path.join(PATH, "wafer_"+filename_stub+".xml"), True)

    print "aquiring done"

def read_voltages():

    cmd = "ssh -F /dev/null -x -i ./id_rsa_resetuser resetuser@raspeval-001 -o PubkeyAuthentication=yes /home/pi/voltages/readVoltages"
    data = dict()
    for l in os.popen(cmd).read().rstrip('\n').split('\n'):
        if "voltage" in l:
            prefix = ""
            continue
        if "dac" in l:
            prefix = "DAC_"
            continue
        data.update({ prefix+l.split('\t')[0]: map(float, l.split('\t')[1:]) })

    return data

"""
def store_voltages(filename):

    data = read_voltages()

    with open(filename, "w") as f:

        json.dump(data, f)
"""

if __name__ == "__main__":

    random.seed(1)

    neuron_blacklist = np.loadtxt('blacklist_w{}_h{}.csv'.format(WAFER, HICANN)) # no notion of freq and bkgisi

    hardware = shallow.Hardware(WAFER, HICANN, os.path.join(args.calibpath,'wafer_{0}').format(WAFER), FREQ*1e6, BKGISI)

    hardware.connect()

    # one block per voltage setting
    # think about indexing all spiketrains, maybe just empty for not-scanned drivers

    drivers = args.drivers

    prepare_drivers(drivers)

    reader = NeoHdf5IO(filename=os.path.join(PATH,suffix+".hdf5"))
    # create a new block
    blk = Block(time=time.time(),wafer=WAFER,hicann=HICANN,freq=FREQ,bkgisi=BKGISI)

    for driver in drivers:

        start = time.time()

        data_voltages = read_voltages()
        seg = Segment(driver=driver, voltages=data_voltages)
        blk.segments.append(seg)

        aquire(seg, driver)
        #store_voltages(os.path.join(PATH, "voltages_"+filename_stub+".json"))

        if args.ana:
            ana_defects.ana(seg, plotpath=PATH)

        print "it took {} s".format(time.time()-start)

    reader.write(blk)
