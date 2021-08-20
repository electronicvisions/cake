import numpy
import pysthal
import pylogging
import shallow
import sys
from pyhalco_hicann_v2 import *

pylogging.default_config()

VOL = float(sys.argv[1])
VOH = float(sys.argv[2])

db = pysthal.MagicHardwareDatabase()

hardware = shallow.Hardware(0, 280, None, 100*1e6, 100)
hardware.connect()
hardware.wafer.configure(shallow.VOLVOHHICANNConfigurator(VOL, VOH))

print("+" * 80)
for v, channel in [(VOL, ChannelOnADC(4)), (VOH, ChannelOnADC(1))]:
    cfg = db.get_adc_of_hicann(hardware.hicann.index(), AnalogOnHICANN(0))
    cfg.channel = channel
    recorder = pysthal.AnalogRecorder(cfg)
    recorder.record(1e-3)
    data = recorder.trace()
    recorder.freeHandle()
    mean = numpy.mean(data)
    err = numpy.std(data) / numpy.sqrt(len(data-1))
    print("{}: {:.2f} +- {:.2f} ({:.2f})".format(channel, mean, err, v))
