import numpy as np

import pyhalbe
import pycairo.config.adc

class ADCTrace(object):
    '''Container to store a measured ADC trace.
    
    Automatically converts raw data to voltage and adds time.'''

    def __init__(self, raw_data=None):
        self.raw = None
        self.voltage = None
        self.time = None

        if raw_data is not None:
            self.set_trace(raw_data)

    def set_trace(self, raw_data):
        '''Store measured data.'''

        self.raw = np.array(raw_data)
        self.time = np.arange(len(raw_data))*1e-6
        self.voltage = self._convert_to_voltage(self.raw)

    def _convert_to_voltage(self, raw):
        '''Convert a raw ADC readout value to a voltage.

        Args:
            raw: raw readout from the ADC
        '''

        return (2.3139-0.000821*(raw)+pow(0.000239*(raw),2)-pow(0.0002*(raw),3))

    def get_mean(self):
        return np.mean(self.voltage)

    def get_min(self):
        return np.min(self.voltage)

    def get_max(self):
        return np.max(self.voltage)


class ADCInterface():
    '''This class encapsulates the methods to readout from the ADC, as well as higher level methods to read the spiking frequency of signal.'''

    def __init__(self):
        self.sampletime = pycairo.config.adc.SampleTime()

    def read_adc(self, sample_time):
        '''Start the acquisition from the ADC and return times and voltages.

        Args:
            sample_time: The desired sample time in us
        '''

        adc = pyhalbe.Handle.ADC()
        cfg = pyhalbe.ADC.Config(sample_time,
                                 pycairo.config.adc.INPUT_CHANNEL,
                                 pyhalbe.ADC.TriggerChannel(0))
        pyhalbe.ADC.config(adc, cfg)
        pyhalbe.ADC.trigger_now(adc)
        raw = pyhalbe.ADC.get_trace(adc)
        return ADCTrace(raw)

    def get_mean(self):
        '''Get the mean value of the signal for 100us.'''

        trace = self.read_adc(self.sampletime.mean)
        return trace.get_mean()

    def get_min(self):
        '''Get the minimum value of the signal for 100us.'''

        trace = self.read_adc(self.sampletime.min)
        return trace.get_min()

    def get_max(self):
        '''Get the maximum value of the signal for 100us.'''

        trace = self.read_adc(self.sampletime.max)
        return trace.get_max()

    def get_freq(self):
        '''Get the spiking frequency of the signal'''

        spikes = self.get_spikes()

        # Get freq
        ISI = spikes[1:] - spikes[:-1]

        if (len(ISI) == 0) or (np.mean(ISI) == 0):
            return 0
        else:
            return 1.0/np.mean(ISI)*1e6

    def get_spikes(self):
        '''Get the spikes from the signal'''

        trace = self.read_adc(self.sampletime.spikes)
        t,v = trace.time, trace.voltage
        v = np.array(v)
        # Derivative of voltages
        dv = v[1:] - v[:-1]
        # Take average over 3 to reduce noise and increase spikes 
        smooth_dv = dv[:-2] + dv[1:-1] + dv[2:]
        threshhold = -2.5 * np.std(smooth_dv)

        # Detect positions of spikes
        tmp = smooth_dv < threshhold
        pos = np.logical_and(tmp[1:] != tmp[:-1], tmp[1:])
        spikes = t[1:-2][pos]

        isi = spikes[1:] - spikes[:-1]
        if max(isi) / min(isi) > 1.7 or np.std(isi)/np.mean(isi) >= 0.1:
            # write raw data to file for debug purposes
            from time import time
            filename = "get_spikes_" + str(time()) + ".npy"
            np.save(filename, (t,v))

            x = smooth_dv
            print "Stored rawdata to:", filename
            print "min, max, len", min(x), max(x), len(x)
            print "mean, std", np.mean(x), np.std(x)

            print "min, max, len", min(isi), max(isi), len(isi)
            print "--> mean, std", np.mean(isi), np.std(isi)

        return spikes


    def get_spikes_bio(self):
        '''Get the spikes from the signal and convert to bio domain'''

        spikes = self.get_spikes()
        for i,item in enumerate(spikes):
            spikes[i] = spikes[i]/1e6*1e7

        return spikes

    def get_freq_bio(self):
        '''Get the frequency in the bio domain'''

        freq = self.get_freq()
        return freq/1e4

    def adc_sta(self, period):
        '''Perform Spike-Triggered Averaging

        Args:
            period: The period of the stimulus
        '''

        mean_v_all = []
        mean_v = []

        t,v = self.start_and_read_adc(self.sampletime.spikes)

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
