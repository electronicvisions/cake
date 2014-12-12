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

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from scipy import interpolate
from scipy.interpolate import griddata

import json

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
    parser.add_argument('--plotfilename', type=str, help="without suffix")
    parser.add_argument('--show', action="store_true", default=False)
    parser.add_argument('--verbose', action="store_true", default=False)
    parser.add_argument('--ignore_neurons', type=int, nargs="*", default=[])
    parser.add_argument('--clim', type=int, nargs=2, default=None)
    parser.add_argument('--diff', action="store_true", default=False)
    parser.add_argument('--print_bad', action="store_true", default=False)
    args = parser.parse_args()

    xs = []
    ys = []
    freqs = []
    bkgisis = []
    Vccas = []
    Vdllres = []

    l_n_good_drv = []

    if args.diff and args.files_are_plotdata:
        print "options diff and files_are_plotdata clash"
        exit(1)

    if args.diff:
        if len(args.files) != 2:
            print "diff needs exactly two files"
            exit(1)

    to_be_diffed = [ [], [] ]

    # one file per voltage setting
    for n, f in enumerate(args.files):

        print "processing {} {}".format(n, f.name)

        fdir, filename = os.path.split(os.path.abspath(f.name))

        ana_file = os.path.join(fdir, os.path.splitext(filename)[0]+".json")

        if not args.diff and not args.files_are_plotdata:

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

            try:
                #print blk.annotations['shared_parameters']
                pass
            except KeyError:
                print "no shared parameters stored in block annotations"
                pass

        except:
            print f.name, "is bad, continue with next"
            continue

        if args.files_are_plotdata:
            try:
                with open(ana_file) as fana:
                    ana_results = json.load(fana)
            except IOError as e:
                print ana_file, "is bad, continue with next"
                continue

        #print "blk.annotations", blk.annotations

        bkgisi = blk.annotations["bkgisi"]
        freq   = blk.annotations["freq"]

        bkgisis.append(bkgisi)
        freqs.append(freq)
        Vccas.append(blk.annotations["shared_parameters"][0]["V_ccas"])
        Vdllres.append(blk.annotations["shared_parameters"][0]["V_dllres"])

        if args.files_are_plotdata == False:

            segments = blk.segments

            addr_n_silent = defaultdict(int)
            addr_n_bad = defaultdict(int)

            n_good_drv = 0

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

            for seg in segments:
                driver = seg.annotations["driver"]

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
                    if args.verbose:
                        print "driver {:03d} is good".format(driver)
                    if not args.diff:
                        try:
                            os.symlink("../driver_{:03d}.pdf".format(driver), os.path.join(fdir,"good/driver_{:03d}.pdf".format(driver)))
                        except OSError as e:
                            #print e
                            pass
                else:
                    is_good.append(False)
                    if args.verbose:
                        print "driver {:03d} is bad".format(driver)
                    if not args.diff:
                        try:
                            os.symlink("../driver_{:03d}.pdf".format(driver), os.path.join(fdir,"bad/driver_{:03d}.pdf".format(driver)))
                        except OSError as e:
                            #print e
                            pass

                to_be_diffed[n].append([fdir, driver, is_good[-1], green_to_reds[-1], vols_board_0[-1], vohs_board_0[-1], vols_board_1[-1], vohs_board_1[-1], vols_DAC_board_0[-1], vohs_DAC_board_0[-1], vols_DAC_board_1[-1], vohs_DAC_board_1[-1]])

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

            print fdir, n_good_drv, mean_vols_board_0, std_vols_board_0, mean_vols_board_1, std_vols_board_1, mean_vohs_board_0, std_vohs_board_0, mean_vohs_board_1, std_vohs_board_1
            if args.verbose:
                print "DAC", mean_vols_DAC_board_0, std_vols_DAC_board_0, mean_vols_DAC_board_1, std_vols_DAC_board_1, mean_vohs_DAC_board_0, std_vohs_DAC_board_0, mean_vohs_DAC_board_1, std_vohs_DAC_board_1

            xs.append(np.mean([mean_vols_board_0, mean_vols_board_1]))
            ys.append(np.mean([mean_vohs_board_0, mean_vohs_board_1]))
            l_n_good_drv.append(n_good_drv)

            print "for plot", xs[-1], ys[-1], l_n_good_drv[-1], freqs[-1], bkgisis[-1], Vccas[-1], Vdllres[-1]

            ana_results = {"VOL":xs[-1], "VOH":ys[-1], "n_good_driver" : l_n_good_drv[-1]}

            with open(ana_file, 'w') as fana:
                json.dump(ana_results, fana)

            #blk.annotations["ana"] = ana_results
            #reader.save(blk) # takes 2 minutes -> too long

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

            xs.append(ana_results["VOL"])
            ys.append(ana_results["VOH"])
            l_n_good_drv.append(ana_results["n_good_driver"])

    if args.diff:
        for entry_0, entry_1 in zip(to_be_diffed[0], to_be_diffed[1]):
            if entry_0[2] != entry_1[2]:
                print entry_0[1:]
                print entry_1[1:]
                print

    if args.print_bad:

        print "bad: "
        # reuse list for diffing
        bad = sorted([entry_0[1] for entry_0 in to_be_diffed[0] if entry_0[2] == False])
        print "[" + ", ".join([str(b) for b in bad]) +"]"

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

        if args.show:
            plt.show()
        if args.plotfilename:
            fig.savefig(args.plotfilename+".pdf")

        plt.close()

        fig, ax = plt.subplots(figsize=(12, 10))
        margins={"left":0.11, "right":0.95, "top":0.95, "bottom":0.11}
        plt.subplots_adjust(**margins)

        mean_vo = [(x+y)/2 for x,y in zip(xs,ys)]
        diff_vo = [(y-x) for x,y in zip(xs,ys)]

        plt.scatter(diff_vo, mean_vo, c=l_n_good_drv,s=500, cmap=plt.cm.jet)

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

        plt.grid(True)

        if args.clim:
            plt.clim(*args.clim)

        if args.plotfilename:
            fig.savefig(args.plotfilename+"_mean_vs_diff.pdf")

        if args.plotfilename:

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
