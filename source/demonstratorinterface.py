## @package demonstratorinterface
# Python interface for Demonstrator platform

# vim: set noexpandtab:

# Import

import xmloutputconfigure as XMLOC
import os
import numpy
import pylab
import sys
import time
import shlex
import subprocess
import StringIO

debug_out = True

# Configuration thrown during a communication Error
class Tests2Exception(Exception):
	pass

## Class DemonstratorInterface
class DemonstratorInterface:

	## The constructor
	# @param tm_path The path of the tests2 executable
	# @param xmlfile_path The path of the xml file
	# @param mode The access mode to the hardware
	# @param h_chain The number of hicanns in the chain
	# @param h_id The id of the current hicann
	# @param jtag_speed The speed of the jtag interface
	def __init__(self,tm_path,xmlfile_path,mode,h_chain,h_id,setup_ip, tests2_args):

		# Ip and port of the setup
		self.ip = list(setup_ip)
		
		# Test mode path
		self.tm_path = tm_path

		# XML comfiguration file path
		self.xmlfile = os.path.join(xmlfile_path, 'FGparam.xml')

		# Access mode, see test2 -h for more informations
		self.tests2_cmd = ["./tests2", '-kill', mode, h_chain, h_id,
				"-ip", setup_ip[0], "-jp", setup_ip[1] ] + tests2_args

		# List of voltages
		self.voltages = ['EL','Vreset','Vt','Vsyni','Vsynx','Vexp','tausyni','tausynx','Esyni','Esynx', "V"]
   
		# Neuron parameters dictionnary with default parameters
		self.fg_p = {'EL':300,
		'Vt':1000,
		'Vexp':300,
		'Vreset':285,
		'Esynx':511,
		'Esyni':511,
		'Vsynx':0,
		'Vsyni':0,
		'tausynx':511,
		'tausyni':511,
		'Iconvi':0,
		'Ibexp':0,
		'Ifireb':0,
		'IgLadapt':0,
		'gL':163,
		'tauref':2000,
		'Iradapt':511,
		'Iconvx':0,
		'Iintbbx':0,
		'Iintbbi':0,
		'Irexp':300,
		'Ispikeamp':1023
		}

		self.lock_hardware()
	
		# Initialize XML output
		self.xmloc = XMLOC.XMLOutput(os.path.dirname(self.xmlfile))
      
		# Floating gate parameters
		self.res_fg = 1023
		self.max_v = 1800 # mV
		self.max_i = 2500 # nA
	
	def call_s2ctrl(self, arg):
		cmd = ["./s2ctrl.py", arg] + self.ip
		p = subprocess.Popen(cmd, cwd=self.tm_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		print arg
		print p.pid
		stdout, stderr = p.communicate()
		print p.returncode
		if (p.returncode != 0):
			print stdout
			raise Tests2Exception("Communication with hardware failed")
		else:
			print stdout, stderr
			print "Hardware " + arg + "ed"

	def unlock_hardware(self):
		self.call_s2ctrl("unlock")

	def lock_hardware(self):
		self.call_s2ctrl("lock")

	## Configure hardware by starting the tm_calibration testmode
	def configure_hardware(self, keys):
		cmd = self.tests2_cmd + ['-m', 'tm_calibration', '-c', self.xmlfile]
		self.call_tests2(cmd, keys)
		
	def call_testmode(self, testmode, keys, extra_args = []):
		assert keys[-1] == 'x'
		cmd = self.tests2_cmd + ['-m', testmode] + extra_args
		self.call_tests2(cmd, keys)
		
	def call_tests2(self, cmd, keys = None):
		assert not isinstance(cmd, basestring)
		if debug_out:
			print cmd
			print " ".join(cmd), "<", r"\n".join(keys if keys else [])
		p = subprocess.Popen(cmd, cwd=self.tm_path,
		                     stdout=subprocess.PIPE, stdin=subprocess.PIPE)
		# Read pipe, print error to console and (hopefully) die
		stdout, stderr = p.communicate("\n".join(keys))
		if (p.returncode == 126):
			print stdout 
			raise Tests2Exception("Communication with hardware failed")
		print "  -->", p.returncode
			
	## Convert voltage to floating gates value
	# @param value The value to convert
	# @param parameter The parameter to convert
	def convert_fg(self,value,parameter):
		if (parameter in self.voltages):
			fgvalue = float(value)/self.max_v*self.res_fg
		else:
			fgvalue = float(value)/self.max_i*self.res_fg
		
		# Check if FG resolution is not excedeed	
		if (fgvalue < 0):
			fgvalue = 0
		if (fgvalue > self.res_fg):
			fgvalue = self.res_fg
			
		return int(fgvalue)
		
	## Convert floating gates value back, for voltages
	# @param fgvalue The value to convert
	def revert_fg(self,fgvalue): 
		value = float(fgvalue)*self.max_v/self.res_fg
		return value
		
	## Convert floating gates value back, for currents
	# @param fgvalue The value to convert
	def revert_fgi(self,fgvalue): 
		value = float(fgvalue)*self.max_i/self.res_fg
		return value
		
	## Create XML configuration file for configuring one chip with calibration
	# @param parameters The neuron parameters
	# @param neuron_index The neuron index
	def export_xml_configure(self,parameters,neuron_index): 
	
		param_g = {
				'Vgmax' : 870, # mV
		}
		param_g.update(parameters[0])

		Vgmax = self.convert_fg(param_g['Vgmax'], "V")
		# Create global parameter line
		p_global = (self.convert_fg(param_g['Vreset'],'Vreset'),1023,400,1023,
			0,1023,0,1023,
			0,Vgmax,0,Vgmax,
			0,Vgmax,0,Vgmax,
			0,0,0,0,
			0,0,0,800)

		# Create arrays for each part of the chip
		n_index_bot_left = []
		n_index_bot_right = []
		n_index_top_left = []
		n_index_top_right = []

		parameters_bot_left = []
		parameters_top_left = []
		parameters_bot_right = []
		parameters_top_right = []

		# Split index and parameters for each part of the chip
		for i,item in enumerate(neuron_index):

			# Top Left
			if (item >= 0 and item < 128):
				n_index_top_left.append(item)
				parameters_top_left.append(parameters[i])

			# Top Right
			if (item >= 128 and item < 256):
				n_index_top_right.append(item-128)
				parameters_top_right.append(parameters[i])

			# Bottom Left
			if (item >= 256 and item < 384):
				n_index_bot_left.append(item-256)
				parameters_bot_left.append(parameters[i])

			# Bottom Right
			if (item >= 384 and item < 512):
				n_index_bot_right.append(item-384)
				parameters_bot_right.append(parameters[i])

		# Prepare arrays for each part of the chip
		p_array_bot_left = self.create_xml_array(n_index_bot_left, parameters_bot_left, 'Left')
		p_array_top_left = self.create_xml_array(n_index_top_left, parameters_top_left, 'Left')
		p_array_bot_right = self.create_xml_array(n_index_bot_right, parameters_bot_right, 'Right')
		p_array_top_right = self.create_xml_array(n_index_top_right, parameters_top_right, 'Right')
		
		# Create XML file
		self.xmloc.create_xml(p_array_top_left,p_array_top_right,p_array_bot_left,p_array_bot_right,p_global,n_index_top_left,n_index_top_right,n_index_bot_left,n_index_bot_right)

	## Calculate arrays for each neuron
	# @param n_index The neuron index
	# @param parameters The neuron parameters
	# @param side The side of the chip, can be "Left" or "Right"
	def create_xml_array(self,n_index,parameters,side):

		# Output array
		pArray = []

		# For left side
		if (side == 'Left'):
			for i,item in enumerate(n_index):
				
				# Take parameters from current neuron
				self.fg_p = parameters[i]
				
				# Create line
				pLine = (0, # Unused
				self.convert_fg(self.fg_p['expAct'],'expAct'), #1
				0, # Unused
				self.convert_fg(self.fg_p['gsyni'],'gsyni'), # 3
				0, # Unused
				1023, # I_fire 5
				self.convert_fg(self.fg_p['EL'],'EL'),
				self.convert_fg(self.fg_p['b'],'b'), #7
				self.convert_fg(self.fg_p.get("Vsyni", 900), "V"), # V_synx
				self.convert_fg(self.fg_p['a'],'a'), #9
				self.convert_fg(self.fg_p['tausyni'],'tausyni'),
				self.convert_fg(self.fg_p['gL'],'gL'), # 11
				self.convert_fg(self.fg_p['Vt'],'Vt'),
				self.convert_fg(self.fg_p['tauref'],'tauref'), # 13
				self.convert_fg(self.fg_p['tausynx'],'tausynx'),
				self.convert_fg(self.fg_p['tw'],'tw'), # 15
				self.convert_fg(self.fg_p['Esynx'],'Esynx'),
				self.convert_fg(self.fg_p['gsynx'],'gsynx'), #17
				self.convert_fg(self.fg_p['Esyni'],'Esyni'),
				self.convert_fg(self.fg_p['Iintbbx'],'Iintbbx'),
				self.convert_fg(self.fg_p['Vexp'],'Vexp'),
				self.convert_fg(self.fg_p['Iintbbi'],'Iintbbi'),
				self.convert_fg(self.fg_p.get("Vsynx", 900), "V"), # V_synx
				self.convert_fg(self.fg_p['dT'],'dT'))
				
				pArray.append(pLine)
			
		# For the right side
		if (side == 'Right'):
			for i,item in enumerate(n_index):
				
				# Take parameters from current neuron
				self.fg_p = parameters[i]
				
				# Create line
				pLine = (self.convert_fg(self.fg_p['Esynx'],'Esynx'),
				self.convert_fg(self.fg_p['gsyni'],'gsyni'),
				self.convert_fg(self.fg_p['Esyni'],'Esyni'),
				self.convert_fg(self.fg_p['gsynx'],'gsynx'),
				self.convert_fg(self.fg_p['Vexp'],'Vexp'),
				self.convert_fg(self.fg_p['Iintbbx'],'Iintbbx'),
				self.convert_fg(self.fg_p.get("Vsynx", 900), "V"),
				self.convert_fg(self.fg_p['Iintbbi'],'Iintbbi'),
				self.convert_fg(self.fg_p['tausyni'],'tausyni'),
				self.convert_fg(self.fg_p['tauref'],'tauref'),
				self.convert_fg(self.fg_p.get("Vsyni", 900), "V"),
				self.convert_fg(self.fg_p['a'],'a'),
				self.convert_fg(self.fg_p['tausynx'],'tausynx'),
				self.convert_fg(self.fg_p['dT'],'dT'),
				self.convert_fg(self.fg_p['Vt'],'Vt'),
				self.convert_fg(self.fg_p['expAct'],'expAct'),
				self.convert_fg(self.fg_p['EL'],'EL'),
				1023,
				0,
				self.convert_fg(self.fg_p['b'],'b'),
				0,
				self.convert_fg(self.fg_p['gL'],'gL'),
				0,
				self.convert_fg(self.fg_p['tw'],'tw'))
				
				pArray.append(pLine)

		return pArray
	      
	## Erase FGArray
	def erase_fg(self):
		self.configure_hardware(["z","x"])
	
	## Create XML file, erase and write FGarray and read out neuron neuronID
	# @param hicann_index The index of HICANNs
	# @param neuron_index The index of neurons
	# @param parameters The neuron parameters
	# @param option The configuration option. Can be "XML_only" or "configure"
	def send_fg_configure(self, neurons, parameters, option='configure'):
		# print "send_fg_configure(",hicann_index,neuron_index,parameters,option,")"
		assert len(neurons) == len(parameters)

		if (option == 'XML_only'):
			#self.h_id = str(hicann)
			self.export_xml_configure(parameters, neurons)

		else:
			#print 'Prog FG'
			#self.h_id = str(hicann)
			self.export_xml_configure(parameters, neurons)
			self.set_neuron_jitter()
			#self.disable_aout()
					
	## Create XML file, erase and write FGarray and read out neuron neuronID
	# @param hicann_index The index of HICANNs
	# @param neuron_index The index of neurons
	# @param parameters The neuron parameters
	def send_fg_configure_syn(self,hicann_index,neuron_index,parameters):
		self.export_xml_configure(neuron_index,parameters)
		self.erase_fg()
		self.set_neuron_jitter_syn()
		self.set_stimulus(0)

	## Sweep output to a given neuron number
	# @param i The neuron number
	# @param side The side of the chip, can be "top" or "bottom"
	# @param current The current to inject in the neuron, values from 0 to 1023	
	def sweep_neuron(self,i,side='top',current=0):
		keys = ["n", str(i), str(current), "O"]
		if (side == 'top'):
			if (i%2==0):
				keys += ["6", "4"]
			else:
				keys += ["4", "6"]
		if (side == 'bottom'):
			if (i%2==0):
				keys += ["7", "5"]
			else:
				keys += ["5", "7"]
		keys.append("x")
		self.configure_hardware(keys)
		
	## Activate all neurons
	# @param neuron_index The neuron index
	def activate_neurons(self,neuron_index):
		keys = []
		for n in neuron_index:
			keys += ['n', str(n)]
		keys.append("x")
		self.configure_hardware("x")

	## Configure one neuron only
	# @param neuron The neuron to configure
	# @param current The current to inject, values from 0 to 1023
	def set_one_neuron_current(self,neuron,current):
		keys = ["q", str(current), "n", str(neuron)]
		self.call_testmode('tm_neuron', keys, ['-c', self.xmlfile])

	## Disable analog outputs	
	def disable_aout(self):
		self.configure_hardware(["y", "x"])
	
	## Switch analog output between top and bottom
	# @param side The side of the chip, can be "top" or "bottom"
	def switch_aout(self,side):
		f = open(self.tm_path + 'key', 'w')
		if (side == 'top'):
			keys = ["0", "6", "4", "x"]
		if (side == 'bottom'):
			keys = ["0", "7", "5", "x"]
		self.configure_hardware(keys)

	## Set MUX to read correct HICANN
	# @param hicann_number The given HICANN number
	def set_mux(self,hicann_number):

		# Calculate correct MUX to use
		if(hicann_number == 0):
			mux = 0 
		if(hicann_number == 1):
			mux = 5
		if(hicann_number == 2):
			mux = 7 

		keys = ['d', '0', str(mux), 'd', '1', str(mux), 'x']
		self.call_testmode('tmak_iboardv2', keys)
   
	## Set stimulus
	# @param i Set the current stimulus to the value i
	def set_stimulus(self,i):
		# Convert to digital value
		self.configure_hardware(["s", str(i), "x"])
		
	## Set ramp stimulus
	# @param i Set the current stimulus to the value i
	def set_ramp_stimulus(self,i):
		# Convert to digital value
		self.configure_hardware(["q", str(i), "x"])
		
	## Set stimulus
	# @param i Set the current stimulus to the value i
	def set_constant_stimulus(self,i):
		# Convert to digital value
		self.configure_hardware(["s", str(i), "x"])
		
	## Set stimulus in nA
	# @param i Set the current stimulus to the value i, in nA
	def set_stimulus_nA(self,i):
		
		# Calculate FG value
		a = -0.0007
		b = 0.56
		c = 2.94
		i = a*i*i+b*i+c
		
		# Convert to digital value
		self.configure_hardware(["q", str(i), "x"])

	## Programm FGarray
	def set_neuron(self):
		self.configure_hardware(["f", "p", "x"])
		self.reset_pll()
	
	def reset_pll(self):
		self.configure_hardware(["F", "3", "1", "x"])

	## Programm FGarray
	def set_neuron_jitter(self):
		keys = ["f", "p","p","s","0","x"]
		self.configure_hardware(keys)
		self.reset_pll()

	## Programm FGarray
	def set_neuron_fast(self):
		keys = ["f", "p","p","s","0","x"]
		self.configure_hardware(keys)
		self.reset_pll()
		
	## Programm FGarray
	def set_neuron_jitter_syn(self):
		keys = ["f", "p","p","p","p","x"]
		self.configure_hardware(keys)
		self.reset_pll()
	
	## Change the slow switch
	# @param i Value of the slow switch
	# @param neuron The desired neuron
	def switch_radapt_slow(self,i,neuron):
		keys = ["d", str(neuron), "C", str(i), "0", str(i), "0", str(i),
		        "0", "1", "x"]
		self.configure_hardware(keys)
	  
	## Change the fast switch
	# @param i Value of the slow switch
	# @param neuron The desired neuron	  
	def switch_radapt_fast(self,i,neuron):
		keys = [ "d", str(neuron), "C", "0", str(i),"0", str(i), "0", str(i),
		         "1", "x" ]
		self.configure_hardware(keys)
		
	## Init HICANN
	def init_L1(self):
		# Reset JTAG and HICANN
		self.call_testmode('tmak_iboardv2', ['1', '2', 'x'])
     
	## Init iBoard and HICANN
	def init_HW(self):
		
		# Reset JTAG and HICANN
		keys = ['1', '2', '3']
		# Set voltages on iBoard and analog out
		keys += ['7', 'c', 'x']

		self.call_testmode('tmak_iboardv2', keys)
		print "iBoard configured"  
 
		# Clear synapse array
		self.configure_hardware(["0","x"])
		print "Synapse array cleared"
        
		# Clear floating gates array
		self.erase_fg()
		print "FG array cleared"
		
	## Read all spikes
	# @param neuron_index The neuron index
	# @param range The number of neuron circuits to read from at a time
	def read_spikes_freq(self,neuron_index,range=4):
		
		def measure_top_half(neuron_index,range):
			neuron_list, freqs = self.read_spikes_half(neuron_index,range)
			
			freqs = list(freqs)
			for i,item in enumerate(neuron_index):
				if (item in neuron_list):
					pass
				else:
					freqs.insert(i,0)
					
			freqs_new = []
			
			print neuron_index
			print len(freqs), freqs
			print len(neuron_list), neuron_list
			
			for i,item in enumerate(neuron_index):
				
				if (item < 128):
					freqs_new.append(freqs[i])
					
				if (item > 127 and item < 256):
					freqs_new.append(freqs[-i+255+128])
					
			return neuron_index, freqs_new
				
		def measure_bot_half(neuron_index,range):
			neuron_list_temp, freqs = self.read_spikes_half(neuron_index,range)
			# Add 256 to neuron list
			neuron_list = []
			for i in neuron_list_temp:
				neuron_list.append(i + 256)
				
			freqs = list(freqs)
			for i,item in enumerate(neuron_index):
				if (item in neuron_list):
					pass
				else:
					freqs.insert(i,0)
					
			freqs_new = []
			
			print neuron_index
			print len(freqs), freqs
			print len(neuron_list), neuron_list
			
			for i,item in enumerate(neuron_index):
				
				if (item > 255 and item < 384):
					freqs_new.append(freqs[i])
					
				if (item > 383):
					freqs_new.append(freqs[-i+255+128])
					
			return neuron_index, freqs_new
		
		# Measure top half
		if (max(neuron_index) < 256):
			neuron_index, freqs_new = measure_top_half(neuron_index,range)
			
		# Measure bottom half
		if (min(neuron_index) > 255):
			neuron_index, freqs_new = measure_bot_half(neuron_index,range)
			
		# Measure all chip	
		if (max(neuron_index) > 256 and min(neuron_index) < 255):
			# Split index
			neuron_index_top = []
			neuron_index_bottom = []
			for i in neuron_index:
				if (i < 256):
					neuron_index_top.append(i)
				else:
					neuron_index_bottom.append(i)
			
			# Measure
			neuron_list_top, freqs_top = measure_top_half(neuron_index_top,range)
			neuron_list_bottom, freqs_bottom = measure_bot_half(neuron_index_bottom,range)
			
			# Concatenate results
			neuron_list = neuron_list_top + neuron_list_bottom
			freqs_new = freqs_top + freqs_bottom
						
		return neuron_index, freqs_new
		
	## Read spikes from half the chip
	# @param neuron_index The neuron index
	# @param range The number of neuron circuits to read from at a time
	def read_spikes_half(self,neuron_index,range=4):
		
		keys = []
		for n in numpy.arange(min(neuron_index),max(neuron_index),range):
			keys.append('R')
			keys.append(str(n))
			keys.append(str(n+range-1))
		keys.append("x")
		
		# Remove spike file
		os.system("rm " + self.tm_path + 'train')   

		# Launch test mode
		self.configure_hardware(keys)
      
		# Read file
		try:
			data = numpy.genfromtxt(self.tm_path + "train")
		except:
			data = []
		
		# Neuron index
		neuron_list = []
		spikes_lists = []
		
		if (data != []):
		
			# Sort data
			for i in data:
				neuron_number = (-i[1]+7)*64 + i[2]
				
				if (not (neuron_number in neuron_list)):
					neuron_list.append(int(neuron_number))
									
			# Organize spikes lists
			for i in neuron_list:
				spikes_lists.append([])
				
			for i in data:
				neuron_number = (-i[1]+7)*64 + i[2]
				spikes_lists[neuron_list.index(neuron_number)].append(i[0])
					
			# Calc freq for spikes_lists
			freqs = []
			err_freqs = []
			for i,item in enumerate(spikes_lists):
				frequency = self.calc_freq(item)
				if (frequency > 0):
					freqs.append(frequency)
				else:
					freqs.append(0)
			
			# Sort
			neuronsfreq = zip(neuron_list, freqs)
			neuronsfreq.sort()
			neuron_list, freqs = zip(*neuronsfreq)
			
			# Cut to match neuron_index
			measured_freq = []
			for i,item in enumerate(neuron_list):
				if item in neuron_index:
					measured_freq.append(freqs[i])
					
			# Calculate new neuron list
			neuron_list_new = []
			for i in neuron_list:
				if (i > -1 and i < 32):
					neuron_list_new.append(i)
				if (i > 95 and i < 160):
					neuron_list_new.append(i-96+32)
				if (i > 223 and i < 288):
					neuron_list_new.append(i-224+32+64)
				if (i > 351 and i < 416):
					neuron_list_new.append(i-352+32+128)
				if (i > 479 and i < 512):
					neuron_list_new.append(i-480+32+128+64)
					
		else:
			neuron_list_new = []
			freqs = []
								
		return neuron_list_new, freqs
		
	# Read spikes and plot for one half
	def plot_spikes(self,neuron_index,range=4):
		
		# Measure top half
		if (max(neuron_index) < 256):
			neuron_list, spikes = self.plot_spikes_half(neuron_index,range)
			
		# Measure bottom half
		if (min(neuron_index) > 255):
			temp_neuron_list, spikes = self.plot_spikes_half(neuron_index,range)
			# Add 256 to neuron list
			neuron_list = []
			for i in temp_neuron_list:
				neuron_list.append(i+256)
			
		# Measure all chip	
		if (max(neuron_index) > 256 and min(neuron_index) < 255):
			# Split index
			neuron_index_top = []
			neuron_index_bottom = []
			for i in neuron_index:
				if (i < 256):
					neuron_index_top.append(i)
				else:
					neuron_index_bottom.append(i)
			# Measure
			neuron_list_top, spikes_top = self.plot_spikes_half(neuron_index_top,range)
			neuron_list_bottom, spikes_bottom = self.plot_spikes_half(neuron_index_top,range)
			
			# Concatenate results
			new_neuron_list_bottom = []
			for i in neuron_list_bottom:
				new_neuron_list_bottom.append(i+256)
			
			neuron_list = neuron_list_top + new_neuron_list_bottom
			spikes = spikes_top + spikes_bottom
		
		for i,item in enumerate(neuron_list):
			pylab.scatter(spikes[i],neuron_list[i]*numpy.ones(len(spikes[i])),s=3)
		
	## Read spikes and plot for one half
	# @param neuron_index The neuron index
	# @param range The number of neuron circuits to read from at a time
	def plot_spikes_half(self,neuron_index,range=4):
				
		keys = []
		for n in numpy.arange(min(neuron_index),max(neuron_index),range):
			commands.append('R')
			commands.append(str(n))
			commands.append(str(n+range-1))
		keys.append("x")
		
		# Remove spike file
		os.system("rm " + self.tm_path + 'train')      

		# Launch test mode
		self.configure_hardware(keys)
      
		# Read file
		try:
			data = numpy.genfromtxt(self.tm_path + "train")
		except:
			data = []
		
		if (data != []):
							
			# Neuron index
			neuron_list = []
			spikes_lists = []
		
			# Sort data
			for i in data:
				neuron_number = (-i[1]+7)*64 + i[2]
				
				if (not (neuron_number in neuron_list)):
					neuron_list.append(int(neuron_number))
								
			# Organize spikes lists
			for i in neuron_list:
				spikes_lists.append([])
				
			for i in data:
				neuron_number = (-i[1]+7)*64 + i[2]
				spikes_lists[neuron_list.index(neuron_number)].append(i[0]*1e6)	
				
			# Calculate new neuron list
			neuron_list_new = []
			for i in neuron_list:
				if (i > -1 and i < 32):
					neuron_list_new.append(i)
				if (i > 95 and i < 160):
					neuron_list_new.append(i-96+32)
				if (i > 223 and i < 288):
					neuron_list_new.append(i-224+32+64)
				if (i > 351 and i < 416):
					neuron_list_new.append(i-352+32+128)
				if (i > 479 and i < 512):
					neuron_list_new.append(i-480+32+128+64)
			
		return neuron_list_new, spikes_lists
		
	## Activate background event generator
	# @param BEG_status The status of the BEG, can be "ON" or "OFF"
	# @param BEG_type The mode of the BEG, can be "REGULAR" or "POISSON"
	# @param cycles The number of cycles of the BEG
	# @param neuron The neuron to stimulate witht the BEG
	def activate_BEG(self,BEG_status,BEG_type,cycles,neuron):
		
		# Key file
		if (BEG_status == 'ON'):
			if (BEG_type == 'REGULAR'):
				keys = ['f','h',str(neuron),str(cycles),'H',str(cycles),'0']
			if (BEG_type == 'POISSON'):
				keys = ['f','h',str(neuron),str(cycles),'H',str(cycles),'1','100']
		if (BEG_status == 'OFF'):
			keys = ['7']

		# Launch test mode
		self.configure_hardware(keys + ["x"])
		
	## Deactivate background event generator
	def deactivate_BEG(self):
		self.configure_hardware(["0", "x"])
		
	# Activate 1 neuron sampling experiment
	def neuron_sampling(self,neuron,stimulus,delay):
		
		# Key file
		f = open(self.tm_path + 'key', 'w')
		keys = ["o", str(neuron), str(stimulus), str(delay), 'x']
		self.call_testmode('tm_jitter', keys)
				
	## Reset all Rams
	def erase_all(self):
		self.configure_hardware(["0","F","1","2","5","x"])
		
	## Calc ISI
	# @param spikes_list The input spikes list
	def calc_ISI(self,spikes_list):
		ISI = []
		for i in numpy.arange(1,len(spikes_list)):
			ISI.append(spikes_list[i]-spikes_list[i-1])
		
		return ISI
			
	## Calc frequency
	# @param spikes_list The input spikes list
	def calc_freq(self,spikes_list):
		ISI = self.calc_ISI(spikes_list)
		mean_ISI = numpy.mean(ISI)
		if (mean_ISI != 0):
			return 1/mean_ISI
		else:
			return 0
	
	## Calculate standard deviation of the spiking frequencies
	# @param spikes_list The input spikes list
	def calc_std(self,spikes_list):
		ISI = self.calc_ISI(spikes_list)
		freqs = []
		for i in ISI:
			if (i != 0):
				freqs.append(1/i)
		std_freq = numpy.std(freqs)
		return std_freq
      
      
