"""Spike related functions."""

import numpy as np

def spikes_to_frequency(spikes):
    """Calculate the spiking frequency from spikes."""

    # inter spike interval
    isi = spikes[1:] - spikes[:-1]
    if (len(isi) == 0) or (np.mean(isi) == 0):
        return 0
    else:
        return 1./np.mean(isi)
