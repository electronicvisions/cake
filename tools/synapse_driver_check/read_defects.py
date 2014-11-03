#!/usr/bin/env python

import os
import errno
import sys
import shutil

import argparse

import numpy as np

from collections import defaultdict

from neo.core import (Block, Segment, RecordingChannelGroup,
                      RecordingChannel, AnalogSignal, SpikeTrain)

from neo.io import NeoHdf5IO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# http://stackoverflow.com/a/600612/1350789
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=argparse.FileType('r'), nargs="+")
    parser.add_argument('--files_are_plotdata', action="store_true", default=False)
    parser.add_argument('--plotfilename', type=str)
    parser.add_argument('--show', action="store_true", default=False)
    args = parser.parse_args()

    xs = []
    ys = []

    l_n_good_drv = []

    # one file per voltage setting
    for f in args.files:

        if args.files_are_plotdata == False:

            fdir = f.name.split('/')[0]

            if os.path.isdir(os.path.join(fdir, "good")):
                shutil.rmtree(os.path.join(fdir, "good"))
            if os.path.isdir(os.path.join(fdir, "bad")):
                shutil.rmtree(os.path.join(fdir, "bad"))

            mkdir_p(os.path.join(fdir, "good"))
            mkdir_p(os.path.join(fdir, "bad"))

            try:
                reader = NeoHdf5IO(filename=f.name)
                blks = reader.read(lazy=True, cascade='lazy')
                blk = blks[-1]
            except:
                print f.name, "is bad, continue with next"
                continue

            #print "blk.annotations", blk.annotations
            segments = blk.segments

            addr_n_silent = defaultdict(int)
            addr_n_bad = defaultdict(int)

            n_good_drv = 0

            vols_board_0 = []
            vohs_board_0 = []

            vols_board_1 = []
            vohs_board_1 = []

            green_to_reds = []

            is_good = []

            for seg in segments:
                driver = seg.annotations["driver"]

        #        print "driver", driver

        #        if driver != 10:
        #            continue
        #
                addr_result_map = seg.annotations["addr_result_map"]
                voltages = seg.annotations["voltages"]

                vols_board_0.append(voltages["V9"][0])
                vohs_board_0.append(voltages["V10"][0])

                vols_board_1.append(voltages["V9"][1])
                vohs_board_1.append(voltages["V10"][1])

                drv_correct = 0
                drv_incorrect = 0

                for addr, (correct, incorrect, addr_correlation_map) in addr_result_map.iteritems():
                    if addr != 0: # address 0 will always see "false" background events
                        drv_correct += correct
                        drv_incorrect += incorrect

                try:
                    green_to_red = float(drv_correct)/float(drv_incorrect)
                except Exception:
                    if drv_correct != 0:  # it may be that there are no incorrect events at all
                        green_to_red = 10 # -> push green_to_red over threshold in that case

                #print "green_to_red", green_to_red
                green_to_reds.append(green_to_red)

                if green_to_red > 0.8:
                    n_good_drv += 1
                    is_good.append(True)
                    try:
                        os.symlink("../driver_{:03d}.pdf".format(driver), os.path.join(fdir,"good/driver_{:03d}.pdf".format(driver)))
                    except OSError as e:
                        #print e
                        pass
                else:
                    is_good.append(False)
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

            print fdir, n_good_drv, mean_vols_board_0, std_vols_board_0, mean_vols_board_1, std_vols_board_1, mean_vohs_board_0, std_vohs_board_0, mean_vohs_board_1, std_vohs_board_1

            xs.append(np.mean([mean_vols_board_0, mean_vols_board_1]))
            ys.append(np.mean([mean_vohs_board_0, mean_vohs_board_1]))
            l_n_good_drv.append(n_good_drv)

            print "for plot", xs[-1], ys[-1], l_n_good_drv[-1]

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


            sys.stdout.flush()

        else:
            for line in f:
                ls = [float(x) for x in line.split()]
                xs.append(ls[0])
                ys.append(ls[1])
                l_n_good_drv.append(ls[2])


    fig = plt.figure(figsize=(12, 10))
    margins={"left":0.11, "right":0.95, "top":0.95, "bottom":0.11}
    plt.subplots_adjust(**margins)

    #plt.subplot(2, 2, 3)
    plt.scatter(xs,ys,c=l_n_good_drv,s=500)
    #plt.clim(0,45)
    plt.colorbar()

    plt.xlabel("VOL [V]")
    plt.ylabel("VOH [V]")

    if args.show:
        plt.show()
    if args.plotfilename:
        fig.savefig(args.plotfilename)


    #        corr = np.zeros([64, 64])
    #
    #        for addr, (correct, incorrect, addr_correlation_map) in addr_result_map.iteritems():
    #            for o_addr, counts in addr_correlation_map.iteritems():
    #                corr[addr][o_addr] += counts
    #
    #        plt.pcolor(corr)
    #        plt.colorbar()
    #        plt.savefig("driver_corr_{:02d}.pdf".format(driver))
    #        plt.close()

            # addr is bad if no correct spikes _or_ spikes migrated to other addresses
            # but migration registration spoiled by windows because of start offset


#        for addr, n_silent in addr_n_silent.iteritems():
#            print "{:2d}".format(addr), "".join(["#"]*n_silent)
#
#        for addr, n_bad in addr_n_bad.iteritems():
#            print "{:2d}".format(addr), "".join(["#"]*n_bad)
#

            #for spikes in seg.spiketrains:
            #    print spikes.annotations["addr"], spikes.annotations["n_correct"]
            #    print spikes.annotations["addr"], spikes.annotations["n_incorrect"]
