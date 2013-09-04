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

    return spikes


def spikes_to_freqency(spikes):
    """Calculate the spiking frequency from spikes."""

    # inter spike interval
    isi = spikes[1:] - spikes[:-1]
    if (len(isi) == 0) or (np.mean(isi) == 0):
        return 0
    else:
        return 1./np.mean(isi)
