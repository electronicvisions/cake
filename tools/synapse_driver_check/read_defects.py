#!/usr/bin/env python

import argparse

from neo.core import (Block, Segment, RecordingChannelGroup,
                      RecordingChannel, AnalogSignal, SpikeTrain)

from neo.io import NeoHdf5IO

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=argparse.FileType('r'))
    args = parser.parse_args()

    reader = NeoHdf5IO(filename=args.file.name)

    blks = reader.read(lazy=True, cascade='lazy')
    blk = blks[-1]

    print "blk.annotations", blk.annotations
    segments = blk.segments
    for seg in segments:
        print "seg.annotations", seg.annotations
        #for spikes in seg.spiketrains:
        #    print spikes.annotations["addr"], spikes.annotations["n_correct"]
        #    print spikes.annotations["addr"], spikes.annotations["n_incorrect"]

