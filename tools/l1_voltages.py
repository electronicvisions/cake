#!/usr/bin/env python

from subprocess import Popen, PIPE
import re
import time
import numpy as np
import argparse
import sys

# maximum number of iterations for setting a voltage given in volts
MAX_ITERATIONS = 50
MAX_DAC_STEP = 10
# 
V_OL = 9
V_OH = 10

# unit definitions
VOLT = 'volt'
DAC = 'dac'

SSH_CMD = 'ssh -n -F /dev/null -x -i ./id_rsa_resetuser resetuser@raspeval-001 -o PubkeyAuthentication=yes {0}'

def get_voltage(v, unit=VOLT):
    cmd = SSH_CMD.format('/home/pi/voltages/readVoltages')
    proc = Popen(cmd.split(), stdout=PIPE)
    proc.wait()
    data = proc.communicate()[0]

    if unit == VOLT:
        expr = r'^V{0}\s*([0-9\.]+)\s*([0-9\.]+)$'
    else:
        expr = r'^V{0}\s*([0-9]+)\s*([0-9]+)$'

    voltages = np.array(
            re.search(expr.format(v), data, flags=re.M).groups(),
            dtype=np.float
            )

    return voltages

def reset_voltages():

    cmd = SSH_CMD.format('/home/pi/voltages/chVoltage r')
    proc = Popen(cmd.split(), stdout=PIPE)
    proc.wait()

def set_DAC_both(v, new_DAC):

    if new_DAC < 0:
        return

    current_DAC_0 = get_voltage(v, unit=DAC)[0]
    diff_DAC_0 = new_DAC-current_DAC_0

    current_DAC_1 = get_voltage(v, unit=DAC)[1]
    diff_DAC_1 = new_DAC-current_DAC_1

    counter = 0

    while diff_DAC_0 != 0 or diff_DAC_1 != 0:

        if counter > MAX_ITERATIONS:
            break

        set_DAC(0, v, new_DAC, stop_after_first_step=True)
        set_DAC(1, v, new_DAC, stop_after_first_step=True)

        current_DAC_0 = get_voltage(v, unit=DAC)[0]
        diff_DAC_0 = new_DAC-current_DAC_0

        current_DAC_1 = get_voltage(v, unit=DAC)[1]
        diff_DAC_1 = new_DAC-current_DAC_1

        counter += 1

def set_DAC(board, v, new_DAC, stop_after_first_step=False, first_step=True):

    if new_DAC < 0:
        print "negative DAC {} - do nothing".format(new_DAC)
        return

    if new_DAC < 20:
        print "DAC too small {}, set to limit of 20".format(new_DAC)
        new_DAC = 20

    if board not in [0,1]:
        raise Exception("invalid board id")

    current_DAC = get_voltage(v, unit=DAC)[board]
    diff_DAC = new_DAC-current_DAC

    counter = 0

    while diff_DAC != 0:

        if counter > MAX_ITERATIONS:
            break

        if stop_after_first_step and first_step == False:
            print "stopping after first step"
            break

        print "diff_DAC", diff_DAC, "board", board

        if abs(diff_DAC) <= MAX_DAC_STEP:
            act_new_DAC = new_DAC
        else:
            print "difference {} too large, stepping smaller first".format(diff_DAC)
            act_new_DAC = current_DAC + MAX_DAC_STEP*diff_DAC/abs(diff_DAC)

        print "set voltage number {} on board {} from {} to {}".format(v, board, current_DAC, act_new_DAC)

        cmd = SSH_CMD.format('/home/pi/voltages/chVoltage s {0} {1} {2}'.format(v, board, act_new_DAC))
        proc = Popen(cmd.split(), stdout=PIPE)
        proc.wait()

        time.sleep(2)

        v0, v1 = get_voltage(v, unit=VOLT)
        diff_v = v0-v1
        if abs(diff_v) > 0.1:
            reset_voltages()
            raise Exception("large difference of {} V between boards, resetting to default settings".format(diff_v))

        current_DAC = get_voltage(v, unit=DAC)[board]
        diff_DAC = new_DAC-current_DAC

        first_step = False

        counter += 1

    else:

        print "difference between DAC values is zero, nothing to do"

def step_up(v, n=1):
    cmd = SSH_CMD.format('for i in $(seq {0}); do /home/pi/voltages/chVoltage u {1}; done'.format(n, v))
    proc = Popen(cmd.split(), stdout=PIPE)
    proc.wait()

def step_down(v, n=1):
    cmd = SSH_CMD.format('for i in $(seq {0}); do /home/pi/voltages/chVoltage d {1}; done'.format(n, v))
    proc = Popen(cmd.split(), stdout=PIPE)
    proc.wait()

def set_both_voltage(v, new_DAC):
    """
    set DAC of first board and tune second board to match resulting voltage
    """

    ref_board = 0
    other_board = 1

    # first, set both boards to the same DAC value
    set_DAC_both(v, new_DAC)
    time.sleep(5)

    for i in range(MAX_ITERATIONS):

        # floating reference voltage? good idea?
        ref_voltage = get_voltage(v, unit=VOLT)[ref_board]

        # check voltage on second board
        current_voltage = get_voltage(v, unit=VOLT)[other_board]
        current_DAC = get_voltage(v, unit=DAC)[other_board]

        diff = current_voltage - ref_voltage

        print "diff", diff

        if abs(diff) > 0.005:

            if abs(diff) > 0.01:
                step = 4
            elif abs(diff) > 0.005:
                step = 2
            else:
                step = 1

            if diff > 0:
                # increase DAC
                set_DAC(other_board, v, current_DAC + step)
            else:
                # decrease DAC
                set_DAC(other_board, v, current_DAC - step)

            time.sleep(3)

        else:

            break

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('voltage_number', type=int)
    parser.add_argument('new_value', type=float)
    parser.add_argument('--unit', type=str, choices=[VOLT,DAC], default=DAC)
    parser.add_argument('--board', type=int, choices=[0,1,2], default=None)
    args = parser.parse_args()


    if args.unit == DAC:

        if args.board == None:

            set_both_voltage(args.voltage_number, args.new_value)

        else:

            if args.board in [0,1]:
                set_DAC(args.board, args.voltage_number, args.new_value)
            elif args.board == 2:
                set_DAC_both(args.voltage_number, args.new_value)
            else:
                raise RuntimeError("invalid board number {}".format(args.board))

    elif args.unit == VOLT:

        # first have set both board to equal DACs

        target_volt = args.new_value

        slope = 75 / 0.2 # 75 DACs per 200 mV (guess)
        print "slope {}".format(slope)

        for i in range(MAX_ITERATIONS):

            current_volt_board_0, current_volt_board_1 = get_voltage(args.voltage_number, unit=VOLT)
            current_dac_board_0, current_dac_board_1 = get_voltage(args.voltage_number, unit=DAC)

            diff_volt = target_volt - current_volt_board_0

            if abs(diff_volt) < 0.005:
                print "diff volt small enough, done"
                break

            target_DAC = current_dac_board_0 - diff_volt*slope

            print "target_volt {}".format(target_volt)
            print "current_volt_board_0 {}".format(current_volt_board_0)
            print "target_DAC {}".format(target_DAC)
            print "diff volt {}".format(diff_volt)

            if abs(diff_volt) > 0.05:

                print "diff volt too large, going only by step of 0.05 V"

                diff_volt = diff_volt/abs(diff_volt) * 0.05
                target_DAC = current_dac_board_0 - diff_volt*slope
                print "new diff volt {}".format(diff_volt)
                print "new target_DAC {}".format(target_DAC)

            set_DAC_both(args.voltage_number, int(target_DAC))

            time.sleep(2)

        # equalize voltages of both boards with current DAC

        print "start to equalize voltages"

        current_dac_board_0, current_dac_board_1 = get_voltage(args.voltage_number, unit=DAC)
        set_both_voltage(args.voltage_number, current_dac_board_0)

    else:
        raise RuntimeError("invalid unit {}".format(args.unit))

    time.sleep(3)

    print "l1 voltages:",

    for x in get_voltage(args.voltage_number, unit=DAC):
        print x,

    for x in get_voltage(args.voltage_number, unit=VOLT):
        print x,

    print
