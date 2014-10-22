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

from multiprocessing import Pool

def categorize(addr, spikes, start_offset, addr_offset):

    pos = start_offset + addr*addr_offset

    safety = 3

    correct = spikes[np.abs(spikes - pos - addr_offset/2*safety) <= addr_offset/2*safety]
    incorrect = spikes[np.abs(spikes - pos - addr_offset/2*safety) > addr_offset/2*safety]

    return correct, incorrect

def ana(seg, plot=True):

    driver = seg.annotations["driver"]

    print "analyzing data of driver", driver
    #print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    start_offset = seg.annotations["start_offset"]*quantities.s
    addr_offset = seg.annotations["addr_offset"]*quantities.s
    addr_spikes_map = {spikes.annotations["addr"]: spikes.times for spikes in seg.spiketrains}
    addr_neuron_map = seg.annotations["addr_neuron_map"]

    if plot:
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
        right_yticklabels = []

    for i in range(64):

        if plot:
            plt.axhline(i, c='0.8', ls=':')

        spikes = addr_spikes_map[i]
        correct, incorrect = categorize(i, spikes, start_offset, addr_offset)

        spikes.annotations["n_correct"] = len(correct)
        spikes.annotations["n_incorrect"] = len(incorrect)

        if plot:
            right_yticklabels.append("{},{}".format(len(correct),len(incorrect)))
            plt.plot(correct, [i]*len(correct), 'g|')
            plt.plot(incorrect, [i]*len(incorrect), 'r|')

    if plot:

        ax2 = ax1.twinx()
        ax2.set_ylabel("cor,incor")
        plt.ylim((-1, 64))
        plt.yticks(range(0, 64, 1))
        ax2.set_yticklabels(right_yticklabels)
        plt.tick_params(axis='y', which='both', labelsize=5)

        PATH = "./foobarbaz"

        plt.savefig(os.path.join(PATH, "defects_{}.pdf".format(driver)))
        plt.close()

    print "analyzing done"

    return seg

def ana_no_plot(seg):
    return ana(seg, plot=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=argparse.FileType('r'))
    args = parser.parse_args()

    reader = NeoHdf5IO(filename=args.file.name)
    blks = reader.read()

    # take last block from file for now
    blk = blks[-1]
    #print "blk.annotations", blk.annotations

    start = time.time()

    segments = blk.segments

    #p = Pool(4)
    #augmented_segments = p.map(ana_no_plot, segments)

    # one driver per segment
    augmented_segments = [ana_no_plot(seg) for seg in segments]

    for seg in augmented_segments:
        reader.save(seg)

    print "it took {} s".format(time.time()-start)


