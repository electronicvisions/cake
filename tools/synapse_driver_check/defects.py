#! /usr/bin/env python
# -*- coding: utf-8 -*-

from neo.core import (Block, Segment, RecordingChannelGroup,
                      RecordingChannel, AnalogSignal, SpikeTrain)

from neo.io import NeoHdf5IO

import quantities
import os
import time
import pickle
import subprocess
from collections import defaultdict
import numpy as np
import pyhalbe
import pysthal
import pyhalco_hicann_v2 as Coordinate
from pyhalco_common import Enum, iter_all
from pyhalbe.HICANN import neuron_parameter
from pyhalbe.HICANN import shared_parameter
from pycake.helpers.units import Volt

import pylogging
pylogging.default_config(date_format='absolute')
pylogging.set_loglevel(pylogging.get("pycake.calibtic"), pylogging.LogLevel.ERROR)
pylogging.set_loglevel(pylogging.get("pycake.calibtic"), pylogging.LogLevel.TRACE)
pylogging.set_loglevel(pylogging.get("sthal"), pylogging.LogLevel.INFO)
pylogging.set_loglevel(pylogging.get("sthal.HICANNConfigurator.Time"), pylogging.LogLevel.DEBUG)
pylogging.set_loglevel(pylogging.get("Default"), pylogging.LogLevel.INFO)

import shallow
from pycake.helpers.misc import mkdir_p
import glob
import random
import argparse
import ana_defects
import resource
import shutil
from pprint import pprint

def get_neurons(driver):

    if driver < 112:
        neurons = np.arange(256)
    else:
        neurons = np.arange(256, 512)

    bus = shallow.get_bus_from_driver(driver)

    neurons = np.array([n for n in neurons if n not in neuron_blacklist])

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

    print("preparing drivers")

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

    print("done")

def aquire(seg, driver):

    print("aquiring data for driver", driver)

    hardware.clear_routes()
    hardware.clear_spike_trains()

    # Every synapse driver is accessible via a specific bus. Calculate that bus.
    bus = shallow.get_bus_from_driver(driver)

    even, odd = get_neurons(driver)

    recording_links = []  # set?

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
    print(hardware.hicann.synapses[Coordinate.SynapseDriverOnHICANN(Enum(driver))])

    start_offset = 5e-6
    n_input_spikes = args.ninputspikes
    input_spike_isi = args.inputspikeisi

    seg.annotations["start_offset"] = start_offset
    seg.annotations["addr_neuron_map"] = addr_neuron_map
    seg.annotations["n_input_spikes"] = n_input_spikes
    seg.annotations["input_spike_isi"] = input_spike_isi

    addr_offset = 1.2*(n_input_spikes * input_spike_isi)
    seg.annotations["addr_offset"] = addr_offset

    duration = 4e-3

    addr_position_map = {}

    addresses = list(range(max_index+1))

    # reverse
    #addresses = reversed(addresses)

    # shuffle addresses
    #random.shuffle(addresses)

    # remove addresses
    #addresses.remove(32)
    #addresses.remove(33)
    #addresses.remove(34)
    #addresses.remove(35)
    #addresses.remove(36)

    # manually select addresses
    #addresses = [1,2,3,4,5,32,6,7,8,9,10,11,12,13,14,15]

    for n, addr in enumerate(addresses):

        position = start_offset + addr_offset*n

        addr_position_map[addr] = position

        train = np.arange(n_input_spikes) * input_spike_isi + position
        duration=max(train)+0.001
        hardware.add_spike_train(bus, addr, train)

    seg.annotations["addr_position_map"] = addr_position_map

    seg.annotations["duration"] = duration

    #print hardware.hicann

    record_membrane = False

    filename_stub = '_'.join(["bus",str(bus),"hline", str(hline.value()), "vline", str(vline.value()), "drv",str(driver), suffix])

    record_addr = 44

    # params.neuron[NeuronOnHICANN.Enum(addr_neuron_map[record_addr])].V_t = DAC(1023)

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

    for rl, rl_spikes in rl_dict.items():

        for addr in rl_spikes:
            addr_dict[addr].add(rl)

    for addr, rls in addr_dict.items():
        #print "addr {}, rls {}, len(rls) {}".format(addr, rls, len(rls))
        if len(rls) > 1 and addr != 0:
            #raise Exception("WTF")
            print("addr {}, rls {}, len(rls) {}".format(addr, rls, len(rls)))
            print("PROBLEM(?)")

    spikes = np.vstack([hardware.get_spikes(rl) for rl in recording_links])

    print("len(spikes):", len(spikes))

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

    print("aquiring done")

def vt_calib(neuron_blacklist):
    print("Run V_t calibration")
    meassured = []
    for neuron in iter_all(Coordinate.NeuronOnHICANN):
        if int(neuron.toEnum()) in neuron_blacklist:
            continue
        adc = 0
        trace = hardware.record_neuron(neuron, adc, 1e-5)
        v_t = np.mean(trace) + 0.06
        meassured.append((int(neuron.toEnum()), np.mean(trace)))
        params.neuron_parameters[neuron] = {
            neuron_parameter.V_t: Volt(v_t, apply_calibration=True)}

    np.savetxt(os.path.join(PATH, 'vt_calib.dat'), meassured)

    # print params.neuron_parameters
    hardware.set_parameters(params)
    cfg = UpdateParameterDownAndConfigure([neuron_parameter.V_t],
            Coordinate.FGBlockOnHICANN())
    hardware.configure(configurator=cfg)
    print(cfg.readout_values)


def read_voltages(wafer=1):
    """
    Returns:
        dict : {voltage: value}
    """
    # TODO add vertical setup
    db = pysthal.MagicHardwareDatabase()
    tmp = []
    for channel in [Coordinate.ChannelOnADC(4), Coordinate.ChannelOnADC(1)]:
        cfg = db.get_adc_of_hicann(hardware.hicann.index(), Coordinate.AnalogOnHICANN(0))
        cfg.channel = channel
        recorder = pysthal.AnalogRecorder(cfg)
        recorder.record(1e-3)
        data = recorder.trace()
        recorder.freeHandle()
        mean = np.mean(data)
        tmp.append(mean)

    VOL, VOH = tmp
    return {
        'V9' : [VOL, VOL],
        'V10' : [VOH, VOH],
        'DAC_V9' : [int(VOL*1000), int(VOL*1000)],
        'DAC_V10' : [int(VOH*1000), int(VOH*1000)],
    }

    if wafer == 1:

        data = {}

        if not os.path.isfile("id_rsa_resetuser"):
            raise RuntimeError("file \"id_rsa_resetuser\" missing")

        cmd = ["ssh", "-F", "/dev/null", "-x", "-i", "./id_rsa_resetuser", "resetuser@raspeval-001", "-o PubkeyAuthentication=yes", "/home/pi/voltages/readVoltages"]

        cmd_out = subprocess.check_output(cmd)

        for l in cmd_out.rstrip('\n').split('\n'):
            if "voltage" in l:
                prefix = ""
                continue
            if "dac" in l:
                prefix = "DAC_"
                continue
            data.update({ prefix+l.split('\t')[0]: list(map(float, l.split('\t')[1:])) })

        return data

    elif wafer == 3:

        data = {}

        cmd = ["ssh", "pi@magi-09", "/home/pi/systemControlSoftware/reticleStatus r 11"]

        cmd_out = subprocess.check_output(cmd)

        for l in cmd_out.rstrip('\n').split('\n'):

            data.update({ l.split('\t')[0]: float(l.split('\t')[3]) })

        return data

    else:

        print("no voltage information for wafer {}".format(wafer))

        data = {}

        for i in range(0,12):

            data['V{}'.format(i)] = [0.,0.]
            data['DAC_V{}'.format(i)] = [0.,0.]

        return data


"""
def store_voltages(filename):

    data = read_voltages()

    with open(filename, "w") as f:

        json.dump(data, f)
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('hicann', type=int)
    parser.add_argument('--wafer', type=int, default=0)
    parser.add_argument('--freq', type=int, default=100)
    parser.add_argument('--bkgisi', type=int, default=10000)
    parser.add_argument('--calibpath', type=str, default="/wang/data/calibration/")
    parser.add_argument('--extrasuffix', type=str, default=None)
    parser.add_argument('--dumpwafercfg', action="store_true", default=False)
    parser.add_argument('--ana', action="store_true", default=False,
        help="run ana_defects.py on the fly, to categorize correct/incorrect spiketimes")
    parser.add_argument('--drivers', type=int, nargs="+", default=list(range(224)))
    parser.add_argument('--V_ccas', type=int, default=-1)
    parser.add_argument('--V_dllres', type=int, default=-1)
    parser.add_argument('--nooutput', action="store_true", default=False)
    parser.add_argument('--ninputspikes', type=int, default=200)
    parser.add_argument('--inputspikeisi', type=float, default=0.1e-6)
    parser.add_argument('--vol', type=float, required=True)
    parser.add_argument('--voh', type=float, required=True)
    parser.add_argument('--clearfolder', action="store_true", default=False)
    parser.add_argument('--vt_calib', action="store_true")
    parser.add_argument('--use_parrot_calib', action="store_true")
    args = parser.parse_args()

    WAFER = args.wafer
    HICANN = args.hicann
    FREQ = args.freq
    BKGISI = args.bkgisi

    VOL = args.vol
    VOH = args.voh

    suffix='_'.join(["w{}","h{}","f{}","bkgisi{}","Vccas{}","Vdllres{}","ninputspikes{}","inputspikeisi{}"]).format(WAFER,HICANN,FREQ,BKGISI,args.V_ccas,args.V_dllres,args.ninputspikes,args.inputspikeisi)
    if args.extrasuffix:
        suffix += "_"+args.extrasuffix

    PATH = '_'.join(["defects", suffix])
    FINISHED_TAG=os.path.join(PATH, "finished")

    if os.path.exists(PATH):
        if os.path.exists(FINISHED_TAG):
            print("Skiping allready existing meassurement {}".format(PATH))
            exit(0)
        else:
            print("Remove existing result folder {}..".format(PATH))
            for ii in range(5, 0, -1):
                print("in {}s...".format(ii))
                time.sleep(1)
            shutil.rmtree(PATH)
    mkdir_p(PATH)
    print("START")

    from pycake.helpers.units import DAC, Volt, Ampere


    params = shallow.Parameters()
    if args.use_parrot_calib:
        parrot_params = os.path.join(
            args.calibpath, 'parrot_params.pkl')
        parrot_blacklist = os.path.join(
            args.calibpath, 'parrot_blacklist.txt')

        if not os.path.exists(parrot_params):
            print("Parrot calibration not found")
            exit(-1)
        print("Setting parameters from parrot file!")
        with open(parrot_params, 'r') as infile:
            pparams = pickle.load(infile)
        base_parameters = pparams['base_parameters']
        if args.V_ccas >= 0:
            base_parameters[shared_parameter.V_ccas] = DAC(args.V_ccas)
        if args.V_dllres >= 0:
            base_parameters[shared_parameter.V_dllres] = DAC(args.V_dllres)
        params.base_parameters.update(base_parameters)
        params.synapse_driver.gmax_div = DAC(base_parameters['gmax_div'])
        params.neuron_parameters.update(pparams['neuron_parameters'])
        if args.vt_calib:
            v_t = Volt(0.7, apply_calibration=True)
            for nrn in params.neuron_parameters:
                params.neuron_parameters[nrn][neuron_parameter.V_t] = v_t
        pprint({getattr(p, 'name', p): v for p, v in list(params.base_parameters.items())})
        pprint(params.synapse_driver)
        print(params.neuron_parameters)
    else:
        print("Setting default paramters")
        params.base_parameters[neuron_parameter.E_l] = Volt(0.7)
        params.base_parameters[neuron_parameter.V_t] = Volt(0.745)
        params.base_parameters[neuron_parameter.E_synx] = Volt(0.8)
        params.base_parameters[neuron_parameter.E_syni] = Volt(0.6)
        params.base_parameters[neuron_parameter.V_syntcx] = DAC(800)
        params.base_parameters[neuron_parameter.V_syntci] = DAC(800)
        params.base_parameters[neuron_parameter.I_gl] = DAC(0)
        params.base_parameters[shared_parameter.V_reset] = Volt(0.5)
        if args.V_ccas >= 0:
            params.base_parameters[shared_parameter.V_ccas] = DAC(args.V_ccas)
        if args.V_dllres >= 0:
            params.base_parameters[shared_parameter.V_dllres] = DAC(args.V_dllres)

    random.seed(4)  # chosen by fair dice roll.

    if args.use_parrot_calib:
        parrot_blacklist = os.path.join(
            args.calibpath, 'parrot_blacklist.txt')
        if not os.path.exists(parrot_blacklist):
            print("Parrot calibration blacklist not found")
            exit(-1)
        neuron_blacklist = np.loadtxt(parrot_blacklist, dtype=int)
    else:
        neuron_blacklist = np.array([], dtype=int)


    hardware = shallow.Hardware(WAFER, HICANN, args.calibpath, FREQ*1e6, BKGISI)

    fgc = hardware.hicann.floating_gates

    for p in range(0, int(fgc.getNoProgrammingPasses())):
        f_p = fgc.getFGConfig(shallow.Enum(p))
        f_p.fg_bias = 0
        f_p.fg_biasn = 0
        f_p.pulselength = int(f_p.pulselength.to_ulong() * float(FREQ)/100.0)
        fgc.setFGConfig(shallow.Enum(p), f_p)
        print(fgc.getFGConfig(shallow.Enum(p)))

    hardware.connect()



    # one block per voltage setting
    # think about indexing all spiketrains, maybe just empty for not-scanned drivers

    drivers = args.drivers

    prepare_drivers(drivers)

    hardware.set_parameters(params)
    hardware.configure()
    if args.vt_calib:
        vt_calib(neuron_blacklist)

    if not args.nooutput:
        fname =os.path.join(PATH, "defects_data.hdf5")
        print("opening {}".format(fname))
        reader = NeoHdf5IO(filename=fname)
    # create a new block
    blk = Block(time=time.time(),wafer=WAFER,hicann=HICANN,freq=FREQ,bkgisi=BKGISI)

    from shallow import VOLVOHHICANNConfigurator
    hardware.wafer.configure(VOLVOHHICANNConfigurator(VOL, VOH))

    for driver in drivers:

        start = time.time()

        data_voltages = read_voltages(args.wafer)
        seg = Segment(driver=driver, voltages=data_voltages)
        blk.segments.append(seg)

        aquire(seg, driver)
        #store_voltages(os.path.join(PATH, "voltages_"+filename_stub+".json"))

        if args.ana:
            ana_defects.ana(seg, plotpath=PATH)

        print("it took {} s".format(time.time()-start))

    sp = shallow.HICANN.shared_parameter
    shared = {0:{}, 1:{}, 2:{}, 3:{}}
    for block in shallow.Coordinate.iter_all(shallow.Coordinate.FGBlockOnHICANN):
        for name, param in sp.names.items():
            try:
                shared[int(block.toEnum())][name] = fgc.getShared(block, param)
            except IndexError:
                pass

    print(shared)

    blk.annotations['shared_parameters'] = shared

#    for p in range(0,int(fgc.getNoProgrammingPasses())):
#        print fgc.getFGConfig(shallow.Enum(p))

    if not args.nooutput:
        reader.write(blk)
        print("To analyze results call: ./ana_defects.py", fname)

    if not args.nooutput:
        with open(FINISHED_TAG, 'w'):
            pass

#    readout_wafer = pysthal.Wafer()
#
#    HRC = pysthal.HICANNReadoutConfigurator(readout_wafer)
#
#    hardware.wafer.configure(HRC)
#
#    readout_hicann = readout_wafer[shallow.Coordinate.HICANNOnWafer(shallow.Enum(HICANN))]
#
#    print hardware.hicann
#    print readout_hicann
#
#    print hardware.hicann.repeater
#    print readout_hicann.repeater
