from libsp6 import *
from libsp6_spy import *
import matplotlib.pyplot as plt

def convert_to_voltage(raw):
	return (float(raw)-3193.26)/(-2.914)

def start_and_read_adc(sample_time_us):
	io = Vmoduleusb(0,0x4b4,0x1003)
	usb = Vusbmaster(io)
	status = Vusbstatus(usb)
	memory = Vmemory(usb)
	ocpfifo = Vocpfifo(usb)
	#ocpfifo clients
	ocp = Vocpmodule(ocpfifo,0)
	fastadc = Vflyspi_adc(ocp)
	fpga = Vspikey_fpga(ocp)

	fpga.set_Xctrl_reg(0x00000111)
	fpga.set_Mux(1<<31)

	startaddr = 0;
	adc_num_samples = 0

	adc_num_samples = int(float(sample_time_us * 500) / 10.3) + 1;
	print "Num samples in flyspi_adc::run " + str(adc_num_samples)
	print "Startaddr in flyspi_adc::run "  + str(startaddr)
	endaddr = startaddr + adc_num_samples;
	print "Endaddr in flyspi_adc::run " + str(endaddr)

	fastadc.configure(0)
	fastadc.start(startaddr,endaddr)

	data = memory.readBlock(startaddr+0x08000000,adc_num_samples)
	voltages = []
	times = [i*0.0103 for i in range(adc_num_samples*2)]
	for i in range(adc_num_samples):
		value = int(data[i])
		voltages.append(convert_to_voltage((value>>16)&0xfff))
		voltages.append(convert_to_voltage((value)&0xfff))

	return times,voltages

# ---------
#   Main
# ---------

if __name__ == "__main__":

	times,voltages = start_and_read_adc(100)

	plt.plot(times,voltages,'-o',markersize=2)
	plt.xlabel('us')
	plt.ylabel('mV')
	plt.show()
