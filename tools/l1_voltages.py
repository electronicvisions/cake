from subprocess import Popen, PIPE
import re
import time
import numpy as np

# maximum number of iterations for setting a voltage given in volts
MAX_ITERATIONS = 10

# 
V_OL = 9
V_OH = 10

# unit definitions
VOLT = 'volt'
DAC = 'dac'

SSH_CMD = 'ssh -F /dev/null -x -i ./id_rsa_resetuser resetuser@raspeval-001 -o PubkeyAuthentication=yes {0}'

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


def step_up(v, n=1):
    cmd = SSH_CMD.format('for i in $(seq {0}); do /home/pi/voltages/chVoltage u {1}; done'.format(n, v))
    proc = Popen(cmd.split(), stdout=PIPE)
    proc.wait()

def step_down(v, n=1):
    cmd = SSH_CMD.format('for i in $(seq {0}); do /home/pi/voltages/chVoltage d {1}; done'.format(n, v))
    proc = Popen(cmd.split(), stdout=PIPE)
    proc.wait()

def set_voltage(v, target, unit=VOLT):
    if unit == VOLT:
        for i in range(MAX_ITERATIONS):
            curr = np.mean(np.array(get_voltage(v), dtype=np.float))
            diff = target - curr

            n = int(round(np.abs(diff) * 100))
            if n == 0:
                break

            if diff > 0:
                step_up(v, n)
            else:
                step_down(v, n)

            time.sleep(8)
    else:
        curr, _ = get_voltage(v, unit=DAC)
        diff = target - curr
        n = abs(int(round(diff / 2)))
        if diff > 0:
            step_down(v, n)
        else:
            step_up(v, n)
