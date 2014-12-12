#!/usr/bin/env python

import os
import time

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

            if np.abs(t - pos - addr_offset/2*safety) <= addr_offset/2*safety:
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
    parser.add_argument('--plotpath', default=None)
    args = parser.parse_args()

    ############################################################################
    print "opening file"
    start = time.time()
    reader = NeoHdf5IO(filename=args.file.name)
    blks = reader.read()
    print "done"
    print "it took {} s".format(time.time()-start)

    ############################################################################
    print "starting analysis"
    start = time.time()
    # take last block from file for now
    blk = blks[-1]
    #print "blk.annotations", blk.annotations
    segments = blk.segments
    # one driver per segment
    augmented_segments = [ana(seg, plotpath=args.plotpath) for seg in segments]
    print "done"
    print "it took {} s".format(time.time()-start)

    ############################################################################
    print "storing results"
    start = time.time()
    for seg in augmented_segments:
        reader.save(seg)
    print "done"
    print "it took {} s".format(time.time()-start)


