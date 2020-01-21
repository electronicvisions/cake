#!/usr/bin/env python

import optparse as op
import numpy as np
import pyhalbe
import pysthal
from pyhalco_common import Enum
import pyhalco_hicann_v2 as C
import pylogging

parser = op.OptionParser()
parser.add_option("--wnr", default=33, dest="wnr", type="int", help="wafer number")
parser.add_option("--hnr", default=297, dest="hnr", type="int", help="hicann number")
parser.add_option("--anr", default=0, dest="anr", type="int", help="analog out number")
(opts, args) = parser.parse_args()

pylogging.default_config(date_format='absolute')
pylogging.set_loglevel(pylogging.get("sthal"), pylogging.LogLevel.INFO)
pylogging.set_loglevel(pylogging.get("sthal.HICANNConfigurator.Time"), pylogging.LogLevel.DEBUG)
pylogging.set_loglevel(pylogging.get("Default"), pylogging.LogLevel.INFO)

w = pysthal.Wafer(C.Wafer(opts.wnr))
h = w[C.HICANNOnWafer(Enum(opts.hnr))]

#w.force_listen_local(True)

aout = C.AnalogOnHICANN(opts.anr)
nrn = C.NeuronOnHICANN(Enum(opts.anr))

# set reset above threshold
h.floating_gates.setNeuron(nrn, pyhalbe.HICANN.neuron_parameter.E_l, 600)
h.floating_gates.setNeuron(nrn, pyhalbe.HICANN.neuron_parameter.V_t, 900)

# disable synaptic input
h.floating_gates.setNeuron(nrn, pyhalbe.HICANN.neuron_parameter.I_intbbi, 0)
h.floating_gates.setNeuron(nrn, pyhalbe.HICANN.neuron_parameter.I_intbbx, 0)
h.floating_gates.setNeuron(nrn, pyhalbe.HICANN.neuron_parameter.I_convi, 0)
h.floating_gates.setNeuron(nrn, pyhalbe.HICANN.neuron_parameter.I_convx, 0)

# set reversal potential to leak potential
h.floating_gates.setNeuron(nrn, pyhalbe.HICANN.neuron_parameter.E_synx, 600)
h.floating_gates.setNeuron(nrn, pyhalbe.HICANN.neuron_parameter.E_syni, 600)

# set a long membrane time constant
h.floating_gates.setNeuron(nrn, pyhalbe.HICANN.neuron_parameter.I_gl, 100)

h.enable_aout(nrn, aout)

w.connect(pysthal.MagicHardwareDatabase())

cfg = pysthal.HICANNConfigurator()
w.configure(cfg)

recorder = h.analogRecorder(aout)
recorder.setRecordingTime(1e-1)
recorder.record()

v = recorder.trace()
t = recorder.getTimestamps()

np.save("membrane_w{}_h{}_a{}".format(opts.wnr, opts.hnr, opts.anr), (t,v))
np.savez_compressed("membrane_w{}_h{}_a{}".format(opts.wnr, opts.hnr, opts.anr), (t,v))
