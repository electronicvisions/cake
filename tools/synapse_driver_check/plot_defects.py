#!/usr/bin/env python

import os
import errno
import sys
import math

import argparse

import numpy as np

from collections import defaultdict

from neo.core import (Block, Segment, RecordingChannelGroup,
                      RecordingChannel, AnalogSignal, SpikeTrain)

from neo.io import NeoHdf5IO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from scipy import interpolate
from scipy.interpolate import griddata

import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('plotfilename', type=str, help="without suffix")
    parser.add_argument('files', type=argparse.FileType('r'), nargs="+")
    parser.add_argument('--verbose', action="store_true", default=False)
    parser.add_argument('--clim', type=int, nargs=2, default=None)
    args = parser.parse_args()

    xs = []
    ys = []
    x_stds = []
    y_stds = []
    wafers = []
    hicanns = []
    freqs = []
    bkgisis = []
    Vccas = []
    Vdllres = []
    l_n_good_drv = []

    for n, f in enumerate(args.files):

        try:
            with f as fana:
                print "loading {}".format(f.name)
                ana_results = json.load(fana)
        except IOError as e:
            print ana_file, "is bad, continue with next"
            continue

        bkgisis.append(ana_results["bkgisi"])
        freqs.append(ana_results["freq"])
        hicanns.append(ana_results["hicann"])
        wafers.append(ana_results["wafer"])
        Vccas.append(ana_results["V_ccas"])
        Vdllres.append(ana_results["V_dllres"])
        xs.append(ana_results["VOL"])
        ys.append(ana_results["VOH"])

        x_stds.append(ana_results["STD_VOL"])
        y_stds.append(ana_results["STD_VOH"])


        l_n_good_drv.append(ana_results["n_good_driver"])

    if len(xs) > 1:

        fig, ax = plt.subplots(figsize=(12, 10))
        margins={"left":0.11, "right":0.95, "top":0.95, "bottom":0.11}
        plt.subplots_adjust(**margins)

        #plt.subplot(2, 2, 3)
        plt.scatter(xs,ys,c=l_n_good_drv,s=500, cmap=plt.cm.jet)

        for vol, voh, n_good in zip(xs,ys,l_n_good_drv):
            ax.annotate("{}".format(int(n_good)), (vol, voh), ha="center", va="center", size="xx-small")

        #plt.clim(0,45)
        cbar = plt.colorbar()

        plt.xlabel("VOL [V]")
        plt.ylabel("VOH [V]")

        plt.xlim(0.6,1.5)
        plt.ylim(0.6,1.5)

        # Plot your initial diagonal line based on the starting
        # xlims and ylims.

        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        ax.plot((ax.get_xlim()[0]-0.1, ax.get_xlim()[1]), (ax.get_ylim()[0], ax.get_ylim()[1]+0.1), ls="--", c=".3")
        ax.plot((ax.get_xlim()[0]-0.2, ax.get_xlim()[1]), (ax.get_ylim()[0], ax.get_ylim()[1]+0.2), ls="--", c=".3")
        ax.plot((ax.get_xlim()[0]-0.3, ax.get_xlim()[1]), (ax.get_ylim()[0], ax.get_ylim()[1]+0.3), ls="--", c=".3")

        if args.clim:
            plt.clim(*args.clim)

        idx   = np.argsort(np.array(l_n_good_drv))
        xs_sorted = np.array(xs)[idx]
        ys_sorted = np.array(ys)[idx]
        l_n_good_drv_sorted = np.array(l_n_good_drv)[idx]

        # put solutions in text box
        textstr = "\n".join([
            "min: {}".format(int(l_n_good_drv_sorted[0])),
            "at VOL: {:.2f} V, VOH: {:.2f} V".format(xs_sorted[0], ys_sorted[0]),
            "",
            "max: {}".format(int(l_n_good_drv_sorted[-1])),
            "at VOL: {:.2f} V, VOH: {:.2f} V".format(xs_sorted[-1], ys_sorted[-1])
        ])
        props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.5, 0.45, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        majorLocator   = MultipleLocator(0.05)
        majorFormatter = FormatStrFormatter('%.2f')

        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)

        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_major_formatter(majorFormatter)

        plt.grid(True)

        fig.savefig(args.plotfilename+".pdf")

        plt.close()

        fig, ax = plt.subplots(figsize=(12, 10))
        margins={"left":0.11, "right":0.95, "top":0.95, "bottom":0.11}
        plt.subplots_adjust(**margins)

        mean_vo = [(x+y)/2 for x,y in zip(xs,ys)]
        diff_vo = [(y-x) for x,y in zip(xs,ys)]

        size = [math.sqrt(x_std**2+y_std**2)*1e5 for x_std, y_std in zip(x_stds, y_stds)]

        plt.scatter(diff_vo, mean_vo, c=l_n_good_drv, s=size, cmap=plt.cm.jet)

        for vol, voh, n_good in zip(xs,ys,l_n_good_drv):
            mean = (voh+vol)/2
            diff = (voh-vol)
            ax.annotate("{}".format(int(n_good)), (diff, mean), ha="center", va="center", size="xx-small")

        cbar = plt.colorbar()

        plt.xlim(0,0.5)
        # plt.ylim(0.95,1.25)
        plt.ylim(1,1.3)

        plt.xlabel("VOH-VOL [V]")
        plt.ylabel("(VOH+VOL)/2 [V]")
        cbar.set_label("# good drivers")

        plt.grid(True)

        if args.clim:
            plt.clim(*args.clim)

        fig.savefig(args.plotfilename+"_mean_vs_diff.pdf")

        """
        grid_x, grid_y = np.mgrid[min(xs):max(xs):100j, min(ys):max(ys):100j]

        xs_ys = []
        for x,y in zip(xs,ys):
            xs_ys.append([x,y])

        grid_z1 = griddata(xs_ys, l_n_good_drv, (grid_x, grid_y), method='cubic')

        print grid_z1

        plt.imshow(grid_z1.T, origin='lower')

        plt.colorbar()

        plt.savefig(args.plotfilename+"_interpol.pdf")

        plt.close()
        """

        fig, ax = plt.subplots(figsize=(12, 10))
        margins={"left":0.11, "right":0.95, "top":0.95, "bottom":0.11}
        plt.subplots_adjust(**margins)

        plt.plot(freqs, l_n_good_drv, 'bo')
        # plt.xlim(min(freqs)*0.8, max(freqs)*1.1)
        plt.xlim(25,275)
        plt.ylim(min(l_n_good_drv)*0, max(l_n_good_drv)*1.1)

        plt.xlabel("HICANN frequency [MHz]")
        plt.ylabel("# good drivers")

        plt.grid(True)

        fig.savefig(args.plotfilename + "_vs_freq.pdf")

        plt.close()

        fig, ax = plt.subplots(figsize=(12, 10))
        margins={"left":0.11, "right":0.95, "top":0.95, "bottom":0.11}
        plt.subplots_adjust(**margins)

        plt.plot(bkgisis, l_n_good_drv, 'bo')
        plt.xlim(min(bkgisis)*0.9, max(bkgisis)*1.1)
        plt.ylim(min(l_n_good_drv)*0.9, max(l_n_good_drv)*1.1)

        plt.grid(True)

        plt.xlabel("bkg isi [clocks]")
        plt.ylabel("# good drivers")

        fig.savefig(args.plotfilename + "_vs_bkgisi.pdf")

        # ---

        fig, ax = plt.subplots(figsize=(12, 10))
        margins={"left":0.11, "right":0.95, "top":0.95, "bottom":0.11}
        plt.subplots_adjust(**margins)

        plt.plot(Vccas, l_n_good_drv, 'bo')
        plt.xlim(min(Vccas)*0.9, max(Vccas)*1.1)
        plt.ylim(min(l_n_good_drv)*0.9, max(l_n_good_drv)*1.1)

        plt.grid(True)

        plt.xlabel("V_ccas [DAC]")
        plt.ylabel("# good drivers")

        fig.savefig(args.plotfilename + "_vs_Vcca.pdf")

        plt.close()

        # ---

        fig, ax = plt.subplots(figsize=(12, 10))
        margins={"left":0.11, "right":0.95, "top":0.95, "bottom":0.11}
        plt.subplots_adjust(**margins)

        plt.plot(Vdllres, l_n_good_drv, 'bo')
        plt.xlim(min(Vdllres)*0.9, max(Vdllres)*1.1)
        plt.ylim(min(l_n_good_drv)*0.9, max(l_n_good_drv)*1.1)

        plt.grid(True)

        plt.xlabel("V_dllres [DAC]")
        plt.ylabel("# good drivers")

        fig.savefig(args.plotfilename + "_vs_Vdllres.pdf")

        plt.close()

        # ---

        fig, ax = plt.subplots(figsize=(12, 10))
        margins={"left":0.11, "right":0.95, "top":0.95, "bottom":0.11}
        plt.subplots_adjust(**margins)

        N = 10

        for n in xrange(N):

            ra_low = 0.8 + (1.3-0.8)/N*n
            ra_high = 0.8 + (1.3-0.8)/N*(n+1)

            plt.plot([y for x,y in zip(xs,ys) if x > ra_low and x < ra_high] , [n_good for n_good, x in zip(l_n_good_drv,xs) if x > ra_low and x < ra_high], 'o')

        plt.xlim(min(ys)*0.9, max(ys)*1.1)
        if args.clim:
            plt.ylim(*args.clim)

        plt.grid(True)

        plt.xlabel("VOH [V]")
        plt.ylabel("# good drivers")

        fig.savefig(args.plotfilename + "_vs_voh.pdf")

        # ---

        fig, ax = plt.subplots(figsize=(12, 10))
        margins={"left":0.11, "right":0.95, "top":0.95, "bottom":0.11}
        plt.subplots_adjust(**margins)

        N = 20

        for n in xrange(N):

            ra_low = 1 + (1.4-1.0)/N*n
            ra_high = 1 + (1.4-1.0)/N*(n+1)

            plt.plot([x for x,y in zip(xs,ys) if y > ra_low and y < ra_high] , [n_good for n_good, y in zip(l_n_good_drv,ys) if y > ra_low and y < ra_high], 'o')

        plt.xlim(min(xs)*0.9, max(xs)*1.1)
        if args.clim:
            plt.ylim(*args.clim)

        plt.grid(True)

        plt.xlabel("VOL [V]")
        plt.ylabel("# good drivers")

        fig.savefig(args.plotfilename + "_vs_vol.pdf")

        #--------------------------------------------------------------------------------

        fig, ax = plt.subplots(figsize=(12, 10))
        margins={"left":0.11, "right":0.95, "top":0.95, "bottom":0.11}
        plt.subplots_adjust(**margins)

        plt.plot(l_n_good_drv)

        if args.clim:
            plt.ylim(*args.clim)

        plt.grid(True)

        plt.xlabel("index")
        plt.ylabel("# good drivers")

        fig.savefig(args.plotfilename + "_vs_order.pdf")

        #--------------------------------------------------------------------------------

        fig, ax = plt.subplots(figsize=(12, 10))
        margins={"left":0.11, "right":0.95, "top":0.95, "bottom":0.11}
        plt.subplots_adjust(**margins)

        plt.plot(hicanns, l_n_good_drv)

        majorLocator   = MultipleLocator(1)
        majorFormatter = FormatStrFormatter('%.2f')

        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_major_formatter(majorFormatter)

        if args.clim:
            plt.ylim(*args.clim)

        plt.grid(True)

        plt.xlabel("HICANN")
        plt.ylabel("# good drivers")

        fig.savefig(args.plotfilename + "_vs_hicann.pdf")


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
