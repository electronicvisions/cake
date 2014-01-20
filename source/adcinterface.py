## @package adcinterface
# Documentation for ADCInterface

# vim: set noexpandtab:

# Imports
import matplotlib.pyplot as plt
import numpy as np
import pylab

## Definition of the class ADCInterface
# 
# This class encapsulates the methods to readout from the ADC, as well as higher level methods to read the spiking frequency of signal.
class ADCInterface():
	ADC_CHANNEL = 7

	## The constructor
	def __init__(self):
		import pyhalbe
		self.ADC = pyhalbe.ADC
		self.ADC_Handle = pyhalbe.Handle.ADC

	## Convert the readout from the ADC to a voltage
	# @param raw The raw readout from the ADC
	def convert_to_voltage(self, x):
		a = -0.00064866486377
		b = 2.02942967983
		return a*x+b

	## Start the acquisition from the ADC and return times and voltages
	# @param sample_time_us The desired sample time in us
	def start_and_read_adc(self,sample_time_us):
		from pyhalbe import ADC, Handle
		adc = Handle.ADC()
		cfg = ADC.Config(sample_time_us,
				ADC.INPUT_CHANNEL_7,
				ADC.TriggerChannel(0))
		ADC.config(adc, cfg)
		ADC.trigger_now(adc)
		raw = ADC.get_trace(adc)
		v = self.convert_to_voltage(np.array(raw))
		t = np.arange(len(v)) * 10e-6
		return t, v

	## Get the mean value of the signal for 100 us
	def get_mean(self):

		t,v = self.start_and_read_adc(100)
		return np.mean(v)

	## Get the minimum value of the signal for 100 us
	def get_min(self):

		t,v = self.start_and_read_adc(100)
		return np.min(v)

	## Get the maximum value of the signal for 100 us	
	def get_max(self):

		t,v = self.start_and_read_adc(100)
		return np.max(v)

	## Get the spiking frequency of the signal
	def get_freq(self):

		spikes = self.get_spikes()

		# Get freq
		ISI = spikes[1:] - spikes[:-1]

		if (len(ISI) == 0 or np.mean(ISI) == 0):
			return 0
		else:
			return 1.0/np.mean(ISI)*1e6

	## Get the spikes from the signal
	def get_spikes(self):
		from time import time

		t,v = self.start_and_read_adc(10000)
		v= pylab.array(v)
		# Derivative of voltages
		dv = v[1:] - v[:-1]
		# Take average over 3 to reduce noise and increase spikes 
		smooth_dv = dv[:-2] + dv[1:-1] + dv[2:]
		threshhold = -2.5 * pylab.std(smooth_dv)

		# Detect positions of spikes
		tmp = smooth_dv < threshhold
		pos = pylab.logical_and(tmp[1:] != tmp[:-1], tmp[1:])
		spikes = t[1:-2][pos]

		isi = spikes[1:] - spikes[:-1]
		if max(isi) / min(isi) > 1.7 or pylab.std(isi)/pylab.mean(isi) >= 0.1:
			filename = "get_spikes_" + str(time()) + ".npy"
			pylab.np.save(filename, (t,v))

			x = smooth_dv
			print "Stored rawdata to:", filename
			print "min, max, len", min(x), max(x), len(x)
			print "mean, std", pylab.mean(x), pylab.std(x)


			from pprint import pprint
			#pprint(isi)
			print "min, max, len", min(isi), max(isi), len(isi)
			print "--> mean, std", pylab.mean(isi), pylab.std(isi)
			#pylab.plot(t,v,'o-')
			#pylab.show()
			#sys.exit()

		return spikes


	## Get the spikes from the signal and convert to bio domain
	def get_spikes_bio(self):

		spikes = self.get_spikes()
		for i,item in enumerate(spikes):
			spikes[i] = spikes[i]/1e6*1e7

		return spikes

	## Get the frequency in the bio domain
	def get_freq_bio(self):

		freq = self.get_freq()

		return freq/1e4


	## Perform Spike-Triggered Averaging
	# @param period The period of the stimulus
	def adc_sta(self,period):

		mean_v_all = []
		mean_v = []

		for k in range(1):

			t,v = self.start_and_read_adc(10000)

			dt = t[1]-t[0]
			period_pts = int(period/dt)

			shift = int(2e-6/dt)
			t_ref = t[len(t)/2-1]

			v_middle = v[int(len(v)/2-period_pts/2):int(len(v)/2+period_pts/2)]
			t_middle = t[int(len(v)/2-period_pts/2):int(len(v)/2+period_pts/2)]
			t_ref = t_middle[np.where(v_middle==max(v_middle))[0][0]]

			nb_periods = int((len(t)/2-1)/period_pts)
			t_cut_ref = t[int(t_ref/dt)-shift:int(t_ref/dt)-shift + period_pts]

			# For PSPs after middle
			for i in range(nb_periods):

				v_cut = v[int((t_ref+i*period)/dt)-shift:int((t_ref+(i+1)*period)/dt)-shift]
				if (len(v_cut) == period_pts):
					mean_v_all.append(v_cut)

				print len(v_cut)

			# For PSPs before middle
			for i in range(nb_periods):

				v_cut = v[int((t_ref-(i+1)*period)/dt)-shift:int((t_ref-i*period)/dt)-shift]
				if (len(v_cut) == period_pts):
					mean_v_all.append(v_cut)

				print len(v_cut)

		# Calc mean
		for i in range(period_pts):
			temp_array = []
			for j in mean_v_all:
				temp_array.append(j[i])
			mean_v.append(np.mean(temp_array))

		# Shift time
		init_t = t_cut_ref[0]
		for i, item in enumerate(t_cut_ref):
			t_cut_ref[i] = t_cut_ref[i] - init_t

		return t_cut_ref,mean_v
