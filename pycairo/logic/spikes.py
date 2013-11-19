"""Spike related functions."""

import numpy as np


def detect_spikes(time, voltage):
    """Detect spikes from a voltage trace."""

    # make sure we have numpy arrays
    t = np.array(time)
    v = np.array(voltage)

    # Derivative of voltages
    dv = v[1:] - v[:-1]
    # Take average over 3 to reduce noise and increase spikes
    smooth_dv = dv[:-2] + dv[1:-1] + dv[2:]
    threshhold = -2.5 * np.std(smooth_dv)

    # Detect positions of spikes
    tmp = smooth_dv < threshhold
    pos = np.logical_and(tmp[1:] != tmp[:-1], tmp[1:])
    spikes = t[1:-2][pos]

    return tmp


def spikes_to_freqency(spikes):
    """Calculate the spiking frequency from spikes."""

    # inter spike interval
    isi = spikes[1:] - spikes[:-1]
    if (len(isi) == 0) or (np.mean(isi) == 0):
        return 0
    else:
        return 1./np.mean(isi)


def partition_by_2(seq):
    result = []
    for i in range(0,len(seq)-1):
        result.append([seq[i],seq[i+1]])
    return result

def detect_min_pos(time,voltage):
    """Detect minima between spikes from a voltage trace."""
    t = np.array(time)
    v = np.array(voltage)

    spikes = detect_spikes(time,voltage)
    partitions = partition_by_2(spikes)
    # minimums in each partition between spikes
    mins = map(lambda p: p[0] + np.argmin(v[p[0]:p[1]]), partitions)

    return mins
