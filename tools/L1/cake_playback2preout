#!/usr/bin/env python
# coding: utf-8

# # The plan
#
# * route Buffer6 to Driver109 (HICANNv4)
# * connect Driver109 to preout
# * configure Driver109 to listen for L1Address 16/32/48
# * send spikes on same L1Address, mix in other addresses?
# * record triggered ADC

"""
TODO:

- mix in spikes that should not appear

"""

import numpy as np

import time
import commands
import sqlite3

import pyhalbe
import Coordinate
import pysthal
from pycake.helpers.sthal import StHALContainer
from pycake.helpers.TraceAverager import _find_spikes_in_preout
import pycake.config

Enum = Coordinate.Enum


class CubeIBoard(object):
    """Configure iBoard on cube setup.

    Cube setup iBoard needs to be controlled via USB HID,
    which is accessed via SLURM.
    """

    def srun(self, command):
        return commands.getstatusoutput("srun -p wafer {}".format(command))

    def iboard_ctrl(self, args):
        res = self.srun("--chdir /wang/users/mkleider/cube-setup/hmf-fpga/global_src/software/msp430/firmware/python ./iboardCtrl {}".format(args))
        print res[1]
        if res[0] != 0:
            raise RuntimeError

    def set_voltage(self, voltage, value):
        args = "--iboard 2 set_voltage {} {}".format(voltage, value)
        self.iboard_ctrl(args)

    def switch_mux(self, mux):
        args = "--iboard 2 switch_mux {}".format(mux)
        self.iboard_ctrl(args)


class DatabaseConnection(object):
    def __init__(self, filename):
        conn = sqlite3.connect(filename)
        c = conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS measurements
             (vol REAL, voh REAL, pll INTEGER, l1address INTEGER, recvratio REAL)''')
        conn.commit()

        self.connection = conn
        self.cursor = c

    def add_measurement(self, vol, voh, pll, l1address, recvratio):
        """Insert new measurement"""
        self.cursor.execute("INSERT INTO measurements VALUES (?,?,?,?,?)", (vol, voh, pll, l1address, recvratio))
        self.save()

    def does_exist(self, vol, voh, pll, l1address):
        """See if previous measurement exists"""
        self.cursor.execute('SELECT COUNT(*) FROM measurements WHERE vol=? AND voh=? AND pll=? AND l1address=?', (vol, voh, pll, l1address))
        return self.cursor.fetchone()[0] > 0

    def save(self):
        self.connection.commit()

    def close(self):
        self.save()
        self.connection.close()

    def __del__(self):
        self.close()


class VOLVOHHICANNConfigurator(pysthal.HICANNConfigurator):
    """Set VOL/VOH. Works for vertical setup only.

    Usage: sthal.wafer.configure(VOLVOHHICANNConfigurator(VOL, VOH))
    """

    def __init__(self, vol, voh):
        pysthal.HICANNConfigurator.__init__(self)
        self.vol = vol
        self.voh = voh

    def config_fpga(*args):
        pass

    def config(self, fpga, handle, data):
        self.getLogger().info("SETTING VOL: {}, VOH: {}".format(self.vol, self.voh))
        pyhalbe.Support.Power.set_L1_voltages(handle, self.vol, self.voh)


class PreoutTest(object):
    def __init__(self, setup, wafer=None, hicann=None, dbfile="l1preout.db"):
        """Initialize a sthal container.

        Args:
            setup: type of setup, "vertical", "cube", "wafer"
            wafer: wafer id, only applies if setup is "wafer"
            hicann: id for wafer, "first" or "third" for cube
            dbfile: filename of the sqlite results database
        """

        self.setup = setup
        self.wafer = wafer
        self.hicann = hicann

        if setup == "wafer":
            if wafer != None:
                coord_wafer = Coordinate.Wafer(int(wafer))
                coord_hicann = Coordinate.HICANNOnWafer(Enum(int(hicann))) 
            else:
                raise ValueError("wafer is {}, expected integer.".format(wafer))
            self.hicann_version = 2
        elif setup == "cube":
            # cube setup: Wafer(4), HICANN(88) v2, HICANN(89) v4
            coord_wafer = Coordinate.Wafer(4)
            if hicann == "first":
                coord_hicann = Coordinate.HICANNOnWafer(Enum(88))
                self.hicann_version = 2
            if hicann == "third":
                coord_hicann = Coordinate.HICANNOnWafer(Enum(89))
                self.hicann_version = 4
            else:
                raise ValueError("unknown hicann {}, expected first/third".format(hicann))
            self.set_mux()

        elif setup == "vertical":
            # We are on a vertical setup, the Wafer coordinate must be 0, the first HICANN is 280.
            coord_wafer = Coordinate.Wafer(0)
            coord_hicann = Coordinate.HICANNOnWafer(Enum(280))
            self.hicann_version = 4
        else:
            raise ValueError("unknown setup {}".format(setup))

        parameters = {
            "coord_wafer":  coord_wafer,
            "coord_hicann": coord_hicann,
            "bigcap": True,
            "hicann_version": self.hicann_version,
            "speedup": "normal",
            "wafer_cfg" :  None,
            "save_traces" :  False,
            "PLL" : 125e6,
            "fg_biasn" : 0,
        }

        config = pycake.config.Config("random name", parameters)

        self.sthal = StHALContainer(config)
        self.route_link_enable_background()

        # Initialize database connection to store results.
        self.db = DatabaseConnection(dbfile)

    def route_link_enable_background(self):
        if self.hicann_version == 4:
            # We are testing HICANNv4, the left SynapseDriver connected to preout is 109. Use GbitLink 6 because
            coord_link = Coordinate.GbitLinkOnHICANN(6)
            coord_bg = Coordinate.BackgroundGeneratorOnHICANN(6)
            coord_driver = Coordinate.SynapseDriverOnHICANN(Enum(109))
        elif self.hicann_version == 2:
            coord_link = Coordinate.GbitLinkOnHICANN(7)
            coord_bg = Coordinate.BackgroundGeneratorOnHICANN(7)
            coord_driver = Coordinate.SynapseDriverOnHICANN(Enum(111))
        coord_output = coord_bg.toOutputBufferOnHICANN()

        # We need to mix in background generator spikes for locking. Background generator can only send on L1Address(0) in random mode (hardware bug, issue #xxxx).
        generator = self.sthal.hicann.layer1[coord_bg]
        generator.enable(True)
        generator.random(False)
        generator.period(400)
        generator.address(pyhalbe.HICANN.L1Address(0))

        # Merger are set to merge by default. Do not configure them.

        # Set GbitLink to receiving.
        self.sthal.hicann.layer1[coord_link] = pyhalbe.HICANN.GbitLink.Direction.TO_HICANN

        # Route output buffer to synapse driver.
        self.sthal.hicann.route(coord_output, coord_driver)

        # Connect the driver to preout.
        self.sthal.hicann.analog.set_preout(Coordinate.AnalogOnHICANN(0))

    def set_voltages(self, vol, voh):
        """vol and voh in Volt"""
        if self.setup == "cube":
            # cube setup VOL/VOH
            cube = CubeIBoard()
            cube.set_voltage("vol", vol)
            cube.set_voltage("voh", voh)
        elif self.setup == "vertical":
            self.sthal.wafer.configure(VOLVOHHICANNConfigurator(vol, voh))
        else:
            raise NotImplementedError("implement set_voltages for {} setup".format(self.setup))

    def set_mux(self):
        if self.setup == "cube":
            hicann = self.hicann
            if hicann == "first":
                mux = 0
            elif hicann == "second":
                mux = 1
            elif hicann == "third":
                mux = 2
            else:
                raise ValueError("unknown hicann {}, expected first/second/third".format(hicann))

            cube = CubeIBoard()
            cube.switch_mux(mux)

    def sweep(self):

        print "loop pll"

        for pll in [100, 150, 200, 250]:

            print "PLL", pll

            # Use L1Address 16 because the driver part is 0b10 and the synapse part is 0b0000. We want a falling (or raising) edge in the driver part.
            for l1address in [16, 32, 48]:
                self.run_experiment(l1address, pll)
        self.sthal.disconnect()

    def run_experiment(self, l1address, pll):
        """Run experiment and record ADC.

        Loop over common L1 voltage. Store ratio of spikes received."""

        sthal = self.sthal
        db = self.db

        # Manually create spike train.
        dt = 3e-6
        begin = 5e-6
        no_spikes = 500
        spike_times = np.arange(no_spikes) * dt + begin

        self.sthal.setPLL(float(pll)*1e6)

        myaddress = pyhalbe.HICANN.L1Address(l1address)
        print myaddress

        if self.hicann_version == 4:
            coord_link = Coordinate.GbitLinkOnHICANN(6)
            coord_driver = Coordinate.SynapseDriverOnHICANN(Enum(109))
        elif self.hicann_version == 2:
            coord_link = Coordinate.GbitLinkOnHICANN(7)
            coord_driver = Coordinate.SynapseDriverOnHICANN(Enum(111))
        sthal.configure_synapse_driver(coord_driver, myaddress, 2, 0)

        spikes = pysthal.Vector_Spike()
        for t in spike_times:
            spikes.append(pysthal.Spike(myaddress, t))

        sthal.wafer.clearSpikes()
        sthal.hicann.sendSpikes(coord_link, spikes)

        print "write config"
        sthal.write_config()

        # loop over vol/voh
        runtime = spike_times[-1] + begin + 50e-6

        print "loop voltages"

        voltages_can_be_set = True

        for common in np.arange(0.6, 1.2, 0.05):
            halfdiff = 0.15/2.
            vol = common - halfdiff
            voh = common + halfdiff

            if not db.does_exist(vol, voh, pll, l1address):
                # no cached results, run experiment

                try:
                    print "configure vol={}, voh={}".format(vol, voh)
                    self.set_voltages(vol, voh)
                except NotImplementedError as e:
                    print e
                    voltages_can_be_set = False

                # wait for locking
                time.sleep(0.5)

                runner = pysthal.ExperimentRunner(runtime)
                sthal.adc.activateTrigger(runtime)
                sthal.wafer.start(runner)

                if not sthal.adc.hasTriggered():
                    print "ADC did not trigger for vol={}, voh={}.".format(vol, voh)
                else:
                    trace = sthal.adc.trace()
                    #with open("trace.npy", "w") as outfile:
                    #    np.save(outfile, trace)
                    spikes_on_adc = _find_spikes_in_preout(trace)
                    num_spikes_on_adc = len(spikes_on_adc)
                    print("found {} spikes, mean voltage {}".format(num_spikes_on_adc, np.mean(trace)))
                    recvratio = float(num_spikes_on_adc)/no_spikes

                    db.add_measurement(vol, voh, pll, l1address, recvratio)

            if not voltages_can_be_set:
                break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dbname", help="database filename", type=str)
    parser.add_argument("setup", help="cube/vertical/wafer", type=str)
    parser.add_argument("--hicann", help="cube: first/second/third", type=str, default="")
    parser.add_argument("--wafer", type=int, default=None)
    args = parser.parse_args()

    test = PreoutTest(args.setup, hicann=args.hicann, wafer=args.wafer, dbfile=args.dbname)
    test.sweep()
