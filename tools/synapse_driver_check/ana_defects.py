#!/usr/bin/env python

import os
import time
import shutil

import argparse

import resource

from neo.core import (Block, Segment, RecordingChannelGroup,
                      RecordingChannel, AnalogSignal, SpikeTrain)

from neo.io import NeoHdf5IO

import quantities

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from collections import defaultdict

import json

# http://stackoverflow.com/a/600612/1350789
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def categorize(addr, spikes, start_offset, addr_offset):

    start_offset = float(start_offset)
    addr_offset = float(addr_offset)

    safety = 3

    addr_correlation_map = {}

    correct = []
    incorrect = []

    for t in spikes:

        t = float(t)

        for o_addr in range(64):

            pos = start_offset + o_addr*addr_offset

            if np.abs(t - pos) <= addr_offset*safety:
                if o_addr == addr:
                    correct.append(t)
                if o_addr in addr_correlation_map:
                    addr_correlation_map[o_addr] += 1
                else:
                    addr_correlation_map[o_addr] = 1
            else:
                if o_addr == addr:
                    incorrect.append(t)

    return correct, incorrect, addr_correlation_map

def ana(seg, plotpath=None):

    driver = seg.annotations["driver"]

    print "analyzing data of driver", driver
    #print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    start_offset = seg.annotations["start_offset"]*quantities.s
    addr_offset = seg.annotations["addr_offset"]*quantities.s
    addr_spikes_map = {spikes.annotations["addr"]: spikes.times for spikes in seg.spiketrains}
    addr_neuron_map = seg.annotations["addr_neuron_map"]
    duration = seg.annotations["duration"]

    plot = True if plotpath else False

    if plot:
        fig, ax1 = plt.subplots()
        plt.title("Test Result for Synapse Driver {0}".format(driver))
        plt.xlabel("Time [s]")
        plt.ylabel("Source Address (Neuron, OB)")

        plt.grid(False)
        plt.ylim((-1, 64))
        plt.xlim((0, duration))
        plt.yticks(range(0, 64, 1))

        yticklabels=["{:.0f} ({:.0f}, {})".format(yt, addr_neuron_map[yt], int((addr_neuron_map[yt] % 256) / 32)) for yt in ax1.get_yticks()]
        ax1.set_yticklabels(yticklabels)
        plt.tick_params(axis='y', which='both', labelsize=5)
        right_yticklabels = []

    addr_result_map = {}

    n_spikes_in_total = sum([len(spikes) for spikes in addr_spikes_map.values()])

    early_abort = n_spikes_in_total > 30000

    for i in range(64):

        if plot:
            plt.axhline(i, c='0.8', ls=':')

        spikes = addr_spikes_map[i]

        if not early_abort:
            correct, incorrect, addr_correlation_map = categorize(i, spikes, start_offset, addr_offset)
        else:
            correct = []
            incorrect = spikes
            addr_correlation_map = {}

        if plot:
            right_yticklabels.append("{},{}".format(len(correct),len(incorrect)))
            plt.plot(correct, [i]*len(correct), 'g|')
            plt.plot(incorrect, [i]*len(incorrect), 'r|')

        n_correct = len(correct)
        n_incorrect = len(incorrect)

        addr_result_map[i] = (n_correct, n_incorrect, addr_correlation_map)

    seg.annotations["addr_result_map"] = addr_result_map

    if plot:

        ax2 = ax1.twinx()
        ax2.set_ylabel("cor,incor")
        plt.ylim((-1, 64))
        plt.yticks(range(0, 64, 1))
        ax2.set_yticklabels(right_yticklabels)
        plt.tick_params(axis='y', which='both', labelsize=5)

        if plotpath:
            plt.savefig(os.path.join(plotpath, "driver_{:03d}.pdf".format(driver)))
        plt.close()

    print "analyzing done"

    return seg

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=argparse.FileType('r'))
    parser.add_argument('--ignore_neurons', type=int, nargs="*", default=[])
    parser.add_argument('--verbose', action="store_true", default=False)
    args = parser.parse_args()

    fdir, filename = os.path.split(os.path.abspath(args.file.name))

    ############################################################################
    print "opening file"
    start = time.time()
    reader = NeoHdf5IO(filename=args.file.name)
    blks = reader.read()
    print "done"
    print "it took {} s".format(time.time()-start)

    ############################################################################
    print "starting analysis"

    if False:

        start = time.time()
        # take last block from file for now
        blk = blks[-1]
        #print "blk.annotations", blk.annotations
        segments = blk.segments
        # one driver per segment
        augmented_segments = [ana(seg, plotpath=fdir) for seg in segments]
        print "done"
        print "it took {} s".format(time.time()-start)

        ############################################################################
        print "storing results"
        start = time.time()
        for seg in augmented_segments:
            reader.save(seg)
        print "done"
        print "it took {} s".format(time.time()-start)

    ana_file = os.path.join(fdir, os.path.splitext(filename)[0]+".json")

    if os.path.isdir(os.path.join(fdir, "good")):
        shutil.rmtree(os.path.join(fdir, "good"))
    if os.path.isdir(os.path.join(fdir, "bad")):
        shutil.rmtree(os.path.join(fdir, "bad"))

    mkdir_p(os.path.join(fdir, "good"))
    mkdir_p(os.path.join(fdir, "bad"))

    blk = blks[-1]

    bkgisi = blk.annotations["bkgisi"]
    freq   = blk.annotations["freq"]
    wafer   = blk.annotations["wafer"]
    hicann   = blk.annotations["hicann"]

    # use first block, assuming that all FG blocks have same value
    V_ccas = blk.annotations["shared_parameters"][0]["V_ccas"]
    V_dllres = blk.annotations["shared_parameters"][0]["V_dllres"]

    addr_n_silent = defaultdict(int)
    addr_n_bad = defaultdict(int)

    n_good_drv = 0

    # list for results per driver
    vols_board_0 = []
    vohs_board_0 = []

    vols_board_1 = []
    vohs_board_1 = []

    vols_DAC_board_0 = []
    vohs_DAC_board_0 = []

    vols_DAC_board_1 = []
    vohs_DAC_board_1 = []

    green_to_reds = []

    is_good = []

    for seg in blk.segments:

        driver = seg.annotations["driver"]

        if args.verbose:
            print "classifying driver {}".format(driver)

#        print "driver", driver

#        if driver != 10:
#            continue
#
        addr_result_map = seg.annotations["addr_result_map"]
        addr_neuron_map = seg.annotations["addr_neuron_map"]
        voltages = seg.annotations["voltages"]

        vols_board_0.append(voltages["V9"][0])
        vohs_board_0.append(voltages["V10"][0])

        vols_board_1.append(voltages["V9"][1])
        vohs_board_1.append(voltages["V10"][1])

        vols_DAC_board_0.append(voltages["DAC_V9"][0])
        vohs_DAC_board_0.append(voltages["DAC_V10"][0])

        vols_DAC_board_1.append(voltages["DAC_V9"][1])
        vohs_DAC_board_1.append(voltages["DAC_V10"][1])

        drv_correct = 0
        drv_incorrect = 0

        for addr, (correct, incorrect, addr_correlation_map) in addr_result_map.iteritems():
            # address 0 will always see "false" background events
            if addr != 0 and addr_neuron_map[addr] not in args.ignore_neurons:
                drv_correct += correct
                drv_incorrect += incorrect

        threshold = 0.8

        try:
            green_to_red = float(drv_correct)/float(drv_incorrect)
        except Exception:
            if drv_correct != 0:  # it may be that there are no incorrect events at all
                green_to_red = threshold + 1 # -> push green_to_red over threshold in that case

        #print "green_to_red", green_to_red
        green_to_reds.append(green_to_red)

        if green_to_red > threshold:
            n_good_drv += 1
            is_good.append(True)
            if args.verbose:
                print "driver {:03d} is good".format(driver)

            try:
                os.symlink("../driver_{:03d}.pdf".format(driver), os.path.join(fdir,"good/driver_{:03d}.pdf".format(driver)))
            except OSError as e:
                #print e
                pass
        else:
            is_good.append(False)
            if args.verbose:
                print "driver {:03d} is bad".format(driver)

            try:
                os.symlink("../driver_{:03d}.pdf".format(driver), os.path.join(fdir,"bad/driver_{:03d}.pdf".format(driver)))
            except OSError as e:
                #print e
                pass

    mean_vols_board_0 = np.mean(vols_board_0)
    std_vols_board_0  = np.std(vols_board_0)
    mean_vohs_board_0 = np.mean(vohs_board_0)
    std_vohs_board_0  = np.std(vohs_board_0)

    mean_vols_board_1 = np.mean(vols_board_1)
    std_vols_board_1  = np.std(vols_board_1)
    mean_vohs_board_1 = np.mean(vohs_board_1)
    std_vohs_board_1  = np.std(vohs_board_1)

    mean_vols_DAC_board_0 = np.mean(vols_DAC_board_0)
    std_vols_DAC_board_0  = np.std(vols_DAC_board_0)
    mean_vohs_DAC_board_0 = np.mean(vohs_DAC_board_0)
    std_vohs_DAC_board_0  = np.std(vohs_DAC_board_0)

    mean_vols_DAC_board_1 = np.mean(vols_DAC_board_1)
    std_vols_DAC_board_1  = np.std(vols_DAC_board_1)
    mean_vohs_DAC_board_1 = np.mean(vohs_DAC_board_1)
    std_vohs_DAC_board_1  = np.std(vohs_DAC_board_1)

    mean_vol = np.mean([mean_vols_board_0, mean_vols_board_1])
    mean_voh = np.mean([mean_vohs_board_0, mean_vohs_board_1])

    vol_diffs = [(vol_b_1 - vol_b_0)*1000 for vol_b_0, vol_b_1 in zip(vols_board_0, vols_board_1)]
    nbins = 100
    bins = np.linspace(-20, 20, nbins+1)
    plt.hist([vol_diff for vol_diff, good in zip(vol_diffs, is_good) if good], bins, histtype='bar', rwidth=0.8, color='g')
    plt.hist([vol_diff for vol_diff, good in zip(vol_diffs, is_good) if not good], bins, histtype='bar', rwidth=0.6, color='r', alpha=0.8)
    plt.xlabel("VOL board 1 - VOL board 0 [mV]")
    plt.ylabel("#")
    plt.xlim(-20,20)
    plt.savefig(os.path.join(fdir, "vol_diffs.pdf"))
    plt.close()

    voh_diffs = [(voh_b_1 - voh_b_0)*1000 for voh_b_0, voh_b_1 in zip(vohs_board_0, vohs_board_1)]
    nbins = 100
    bins = np.linspace(-20, 20, nbins+1)
    plt.hist([voh_diff for voh_diff, good in zip(voh_diffs, is_good) if good], bins, histtype='bar', rwidth=0.8, color='g')
    plt.hist([voh_diff for voh_diff, good in zip(voh_diffs, is_good) if not good], bins, histtype='bar', rwidth=0.6, color='r', alpha=0.8)
    plt.xlabel("VOH board 1 - VOH board 0 [mV]")
    plt.ylabel("#")
    plt.xlim(-20,20)
    plt.savefig(os.path.join(fdir, "voh_diffs.pdf"))
    plt.close()

    nbins = 100
    bins = np.linspace((mean_vohs_board_0-2*std_vohs_board_0)*1000, (mean_vohs_board_0+2*std_vohs_board_0)*1000, nbins+1)
    plt.hist([voh_board_0 for voh_board_0, good in zip(np.array(vohs_board_0)*1000, is_good) if good], bins, histtype='bar', rwidth=0.8, color='g')
    plt.hist([voh_board_0 for voh_board_0, good in zip(np.array(vohs_board_0)*1000, is_good) if not good], bins, histtype='bar', rwidth=0.6, color='r', alpha=0.8)
    plt.xlabel("VOH board 0 [mV]")
    plt.ylabel("#")
    #plt.xlim(0,1200)
    plt.savefig(os.path.join(fdir, "voh_board_0.pdf"))
    plt.close()

    nbins = 100
    bins = np.linspace((mean_vols_board_0-2*std_vols_board_0)*1000, (mean_vols_board_0+2*std_vols_board_0)*1000, nbins+1)
    plt.hist([vol_board_0 for vol_board_0, good in zip(np.array(vols_board_0)*1000, is_good) if good], bins, histtype='bar', rwidth=0.8, color='g')
    plt.hist([vol_board_0 for vol_board_0, good in zip(np.array(vols_board_0)*1000, is_good) if not good], bins, histtype='bar', rwidth=0.6, color='r', alpha=0.8)
    plt.xlabel("VOL board 0 [mV]")
    plt.ylabel("#")
    #plt.xlim(0,1200)
    plt.savefig(os.path.join(fdir, "vol_board_0.pdf"))
    plt.close()

    nbins = 100
    bins = np.linspace((mean_vohs_board_1-2*std_vohs_board_1)*1000, (mean_vohs_board_1+2*std_vohs_board_1)*1000, nbins+1)
    plt.hist([voh_board_1 for voh_board_1, good in zip(np.array(vohs_board_1)*1000, is_good) if good], bins, histtype='bar', rwidth=0.8, color='g')
    plt.hist([voh_board_1 for voh_board_1, good in zip(np.array(vohs_board_1)*1000, is_good) if not good], bins, histtype='bar', rwidth=0.6, color='r', alpha=0.8)
    plt.xlabel("VOH board 1 [mV]")
    plt.ylabel("#")
    #plt.xlim(0,1200)
    plt.savefig(os.path.join(fdir, "voh_board_1.pdf"))
    plt.close()

    nbins = 100
    bins = np.linspace((mean_vols_board_1-2*std_vols_board_1)*1000, (mean_vols_board_1+2*std_vols_board_1)*1000, nbins+1)
    plt.hist([vol_board_1 for vol_board_1, good in zip(np.array(vols_board_1)*1000, is_good) if good], bins, histtype='bar', rwidth=0.8, color='g')
    plt.hist([vol_board_1 for vol_board_1, good in zip(np.array(vols_board_1)*1000, is_good) if not good], bins, histtype='bar', rwidth=0.6, color='r', alpha=0.8)
    plt.xlabel("VOL board 1 [mV]")
    plt.ylabel("#")
    #plt.xlim(0,1200)
    plt.savefig(os.path.join(fdir, "vol_board_1.pdf"))
    plt.close()

    ana_results = {"VOL":mean_vol, "VOH":mean_voh, "n_good_driver" : n_good_drv, "wafer": wafer, "hicann": hicann, "freq": freq, "bkgisi": bkgisi, "V_ccas" : V_ccas, "V_dllres" : V_dllres}

    with open(ana_file, 'w') as fana:
        json.dump(ana_results, fana)

    #blk.annotations["ana"] = ana_results
    #reader.save(blk) # takes 2 minutes -> too long
