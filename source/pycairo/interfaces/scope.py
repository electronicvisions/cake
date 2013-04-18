## @package scopeinterface
# High-level Python interface for LeCroy scopes

# Imports
import os
import sys
sys.path.append(os.environ['SYMAP2IC_PATH']+ "/components/pynnhw/src/hardware/stage1/pyscope/")
import facetshardwarescope as fhs
import numpy as np
import time

## Classe ScopeInterface
class ScopeInterface:

	## The constructor
	# @param device The IP address of the scope	
	def __init__(self,device):
		self.scope = fhs.FacetsHardwareScope(device)
   
	## Get the whole scope trace
	# @param channel Number of the channel to use on the scope
	def get_scope_value(self,channel):
		table = self.scope.readTrace(channel,"C")
		return table
	  
	## Get the whole scope trace
	# @param channel Number of the channel to use on the scope
	def get_scope_value_average (self,channel):
		
		# Parameters
		averaging = 5
		v_array = []
		v_mean = []
		v_temp = []
		
		# Measurements
		for i in range(averaging):
			t,v = self.scope.readTrace(channel,"C")
			v_array.append(v)
			
		# Averaging
		for i in range(len(v)):
			v_temp = []
			for j in v_array:
				v_temp.append(j[i])	
			v_mean.append(np.mean(v_temp))
		return t,v_mean
      
	## Get mean value from the scope
	# @param channel Number of the channel to use on the scope
	def get_mean (self,channel):
		mean = self.scope.getAvg(channel,"C")
		return mean
		
	## Init scope
	# @param channel Number of the channel to use on the scope 
	def scope_init(self,channel):
		self.scope.setVoltsDiv(channel,"C",50e-3)
		self.scope.setVerticalOffset(channel,-350e-3)

	## Get minimum value from the scope   
	# @param channel Number of the channel to use on the scope
	def get_min (self,channel):
		mini = self.scope.getMin(channel,"C")
		return mini

	## Get mean minimum value from the scope
	# @param channel Number of the channel to use on the scope   
	def get_mean_min (self,channel):
		mini = self.scope.getMeanMin(channel,"C")
		return mini
	  
	## Get minimum value from the scope
	# @param channel Number of the channel to use on the scope   
	def get_base (self,channel):
		base = self.scope.getBase(channel,"C")
		return base

	## Get maximum value from the scope
	# @param channel Number of the channel to use on the scope   
	def get_max (self,channel):
		maxi = self.scope.getMax(channel,"C")
		return maxi
      
	## Set time division
	# @param time Time per divisions   
	def set_time_div(self,time):
		self.scope.setTimeDiv(time)

	## Set volts/division
	# @param channel Number of the channel to use on the scope
	# @params voltsdivs Volts/division   
	def set_volts_divs(self,channel,voltsdiv):
		self.scope.setVoltsDiv(channel,"C",voltsdiv)

	## Set voltage offset
	# @param channel Number of the channel to use on the scope
	# @param offset Value of the voltage offset
	def set_vertical_offset(self,channel,offset):
		self.scope.setVerticalOffset(channel,offset)
   
	## Get frequency from the scope
	# @param channel Number of the channel to use on the scope   
	def get_freq(self,channel):
		freq = self.scope.getFreq(channel,"C")
		return freq
      
	## Get frequency from the scope
	# @param channel Number of the channel to use on the scope   
	def get_avg_freq(self,channel):
		freq_m = []
		for i in range(5):
			freq_m.append(self.scope.getFreq(channel,"C"))
		return np.mean(freq_m)

	## Get mean frequency from the scope
	# @param channel Number of the channel to use on the scope   
	def get_mean_freq(self,channel):
		freq_m = []
		for i in range(5):
			freq_m.append(self.scope.getFreq(channel,"C"))
		return np.mean(freq_m)
      
	## Sets the trigger
	# @param channel Number of the channel to use on the scope
	def set_trig(self, channel):
		self.scope.setTrig(channel,"C")

	## Clear sweeps
	def clear_sweeps(self):
		self.scope.clearSweeps()

	## Fully adjust scope
	# @param channel Number of the channel to use on the scope
	def scope_adjust(self,channel):
      
		# Visualize whole trace 
		self.scope.setTimeDiv(10e-6)   
		self.scope.setVoltsDiv(channel,"C",225e-3)
		self.scope.setVerticalOffset(channel,-900e-3)

		# Wait
		time.sleep(0.5)	
 
		# Adjust resolution and offset to voltage trace
		maxV = self.scope.getMax(channel,"C")
		minV = self.scope.getMin(channel,"C")
		amp = maxV - minV
		mean = self.scope.getAvg(channel,"C")
		
		self.scope.setVoltsDiv(channel,"C",amp/4)
		self.scope.setVerticalOffset(channel,-mean)

		# Set the trigger at mean
		trigLvl = mean
		self.scope.setTrigLevel(channel,"C",trigLvl)
     
	## Fully adjust scope for synaptic input measurements
	# @param channel Number of the channel to use on the scope
	def scope_adjust_syn(self,channel):

		# Visualize whole trace 
		self.scope.setTimeDiv(10e-6)   
		self.scope.setVoltsDiv(channel,"C",225e-3)
		self.scope.setVerticalOffset(channel,-900e-3)

		# Wait
		time.sleep(0.5)	
 
		# Adjust resolution and offset to voltage trace
		maxV = self.scope.getMax(channel,"C")
		minV = self.scope.getMin(channel,"C")
		amp = maxV - minV
		mean = self.scope.getAvg(channel,"C")
		
		self.scope.setVoltsDiv(channel,"C",10e-3)
		self.scope.setVerticalOffset(channel,-mean)

		# Set the trigger at mean
		trigLvl = mean
		self.scope.setTrigLevel(channel,"C",trigLvl)

	## Spike-Triggered Avering with scope
	# @param period The period of the stimulus
	# @param channel Number of the channel to use on the scope
	def scope_sta(self,period,channel):

		# Set time div
		self.set_time_div(8*period)

		mean_v_all = []
		mean_v = []

		for k in range(1):

			t,v = self.get_scope_value(channel)

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
				
				#print len(v_cut)

			# For PSPs before middle
			for i in range(nb_periods):

				v_cut = v[int((t_ref-(i+1)*period)/dt)-shift:int((t_ref-i*period)/dt)-shift]
				if (len(v_cut) == period_pts):
					mean_v_all.append(v_cut)
				
				#print len(v_cut)
				
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
      
	## Adjust trigger to mean value
	# @param channel Number of the channel to use on the scope
	def set_trigger_mean(self,channel):
		mean = self.scope.getAvg(channel,"C")
		trigLvl = mean
		self.scope.setTrigLevel(channel,"C",trigLvl)

	## Set scope range
	# @param timediv
	# @param vdiv
	# @param voffset
	# @param channel Number of the channel to use on the scope
	def set_scope_range(self,timediv,vdiv,voffset,channel):
		self.setTimeDiv(timediv)
		self.setVoltsDiv(channel,"C",vdiv)
		self.setVerticalOffset(channel,voffset)

