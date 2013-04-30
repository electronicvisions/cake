import pymongo

import numpy

import sys
import os
import datetime

import json

from pycairo.config import coordinates

class DatabaseInterface:
    '''Python interface to the calibration database'''

    def __init__(self):
        connection = pymongo.Connection()

        self.WAFER = connection.calibrationDB.WAFER
        self.FPGA = connection.calibrationDB.FPGA
        self.DNC = connection.calibrationDB.DNC
        self.HICANN = connection.calibrationDB.HICANN
        self.ADC_BOARDS = connection.calibrationDB.ADC_BOARDS

        self.collections = [self.WAFER,self.FPGA,self.DNC,self.HICANN]

        self.parameters = ['Esynx','Esyni','gL','Vreset','Vt','EL','tw','Vexp','tausynx','tausyni','dT','gsyn','b','a','tauref']

        self.reticle_map = coordinates.get_reticle_map()
        self.fpga_map = coordinates.get_fpga_map()
        
    def create_db(self,hardware,fpga_number):
        '''Create database with the selected hardware option

        Args:
            hardware: The hardware system to use. Examples: "USB", "WSS"
            fpga_number: In case of a single FPGA system, the logical number of the FPGA board
        '''

        if (hardware == 'USB'):
            nb_wafers = 1
            nb_FPGAs = 1
            nb_DNCs = 1
            nb_HICANNs = 1
        elif (hardware == 'WSS'):
            nb_wafers = 1
            nb_FPGAs = 12
            nb_DNCs = 4
            nb_HICANNs = 8
    
        neurons_per_hicann = 512
        repeaters_labels = ['rc_l','rc_r','rc_tl','rc_bl','rc_tr','rc_br']
        nb_syn_drivers = 112
        lines_per_array = 224
        columns_per_array = 256
        channels_per_FPGA = 4
        hosts_per_FPGA = 2
        ports_per_FPGA = 2
        
        now = datetime.datetime.now()
        date = now.strftime('%Y-%m-%d %H:%M')

        # Insert JSON documents in database
        for w in range(nb_wafers):
            # Create wafer entry
            wafer = {'logicalNumber' : w, 'uniqueId' : w, 'online' : True}
            self.WAFER.insert(wafer)

            # Create FPGA entry
            for f in range(nb_FPGAs): 
                if hardware == 'USB':
                    fpga_id = fpga_number # Unique ID
                else:
                    fpga_id = f

                fpga_channels = []
                for c in range(channels_per_FPGA):
                    fpga_channel = {'logicalNumber': c, 'sink': '0.00.0'}
                    fpga_channels.append(fpga_channel)

                hosts = []
                for h in range(hosts_per_FPGA):
                    host = {'fpgaPort': h, 'ip': coordinates.get_fpga_host_ip(h), 'online': True}
                    hosts.append(host)

                ports = []
                for p in range(ports_per_FPGA):
                    port = {'logicalNumber': p, 'ip': coordinates.get_fpga_ip(fpga_id)}
                    ports.append(port)
                    
                # ADC calibration data
                analogReadout = [{"adc": "0", "channel": -1}, {"adc": "0", "channel": -1}]

                fpga = {'uniqueId': fpga_id,
                        'logicalNumber': fpga_id,
                        'location': self.get_location(fpga_id),
                        'fpgaX': self.get_fpga_xpos(fpga_id),
                        'fpgaY': self.get_fpga_ypos(fpga_id),
                        'online': False,
                        'parent_wafer': w,
                        'locked': False,
                        'host': hosts,
                        'fpgaPort': ports,
                        'fpgaChannel': fpga_channels,
                        'analogReadout': analogReadout}
                self.FPGA.insert(fpga)

                for d in range(nb_DNCs):
                    # Create DNC entry
                    dnc = { 'logicalNumber': fpga_id*nb_DNCs+d,
                            'uniqueId': fpga_id*nb_DNCs+d,
                            'reticleId': self.get_reticle_id(fpga_id,d),
                            'parent_wafer': w,
                            'parent_fpga': fpga_id,
                            'location': '0',
                            'available': False,
                            'locked': False,
                            'dncX': self.get_dnc_xpos(fpga_id,d),
                            'dncY': self.get_dnc_ypos(fpga_id,d),
                            'fpgaDncChannel': d}
                    self.DNC.insert(dnc)

                    for h in range(nb_HICANNs):
                        # Store global switches values
                        global_switches = {'gL': [0,0,0,0],
                                           'gLadapt': [0,0,0,0],
                                           'bigCap': [1,1]}

                        # Create repeaters array. Repeaters block numbers from 0 to 5 : rc_l,rc_r,rc_tl,rc_bl,rc_tr,rc_br
                        repeaters = []
                        for r in repeaters_labels:
                            repeater_block = []
                            if (r == 'rc_l' or r == 'rc_r'):
                                repeaters_per_block = 32
                            else:
                                repeaters_per_block = 64
                            for i in range(repeaters_per_block):
                                repeater = {'logicalNumber': i, 'available': True}
                                repeater_block.append(repeater)

                            repeaters.append(repeater_block)
                        
                        synapse_drivers = []
                        for halves in range(2): # Two halves of the chip
                            syn_driver_half = []
                            for s in range(nb_syn_drivers):
                                synapse_driver = {'logicalNumber': s, 'available': True}
                                syn_driver_half.append(synapse_driver)
                            synapse_drivers.append(syn_driver_half)
                    
                        # Create synapse array
                        synapses = []

                        # Two halves of the chip
                        for halves in range(2):
                            syn_half = []
                            for l in range(lines_per_array):
                                syn_line = []
                                for c in range(columns_per_array):
                                    synapse = {'logicalNumber': c, 'available': True}
                                    syn_line.append(synapse)
                                syn_half.append(syn_line)
                            synapses.append(syn_half)
                    
                        # Create neuron array
                        neurons = []
                        for n in range(neurons_per_hicann):
                            neuron = {'logicalNumber': n, 'calibration_date': date, 'available': True}
                            for k in self.parameters:
                                neuron[k] = []
                                neuron[k + '_dev'] = []
                                neuron[k + '_fit'] = []
                                neuron[k + '_calibrated'] = False
                            neurons.append(neuron)

                        # Create HICANN entry
                        hicann = {'reticleId': self.get_reticle_id(fpga_id,d), 
                                  'reticleX': self.get_reticle_xpos(fpga_id,d,h), 
                                  'reticleY': self.get_reticle_ypos(fpga_id,d,h), 
                                  'hicannX': self.get_hicann_xpos(fpga_id,d,h), 
                                  'hicannY': self.get_hicann_ypos(fpga_id,d,h), 
                                  'configId': self.get_conf_id(fpga_id,d,h),
                                  'uniqueId': fpga_id*nb_HICANNs*nb_DNCs+d*nb_HICANNs+h, 
                                  'dncHicannChannel': h,
                                  'parent_wafer': w,
                                  'parent_fpga': fpga_id,  
                                  'parent_dnc': d, 
                                  'locked': False, 
                                  'available': False,
                                  'calibrated': False, 
                                  'neurons': neurons,
                                  'repeaters': repeaters,
                                  'synapse_drivers': synapse_drivers,
                                  'synapses': synapses,
                                  'global_switches': global_switches}
                        
                        self.HICANN.insert(hicann)

    def insert_wafer(self,number,w_id,online):
        '''Insert a wafer in the database.

        Args:
            number: The logical number of the wafer
            w_id: The unique ID of the wafer
            online: Status of the wafer, can be True or False
        '''

        wafer = {'logicalNumber': number, 'uniqueId': w_id, 'online': online}
        self.WAFER.insert(wafer)
    
    def insert_adc_board(self,serien_no,calib):
        if not isinstance(serien_no, basestring):
            raise TypeError("serien_no must be a string")
        if not len(calib) == 8:
            raise ValueError("Calibrations for each mux must be provided")
        if not all( [ x.has_key["model"] and x.has_key["coefficients"] for x in calib ] ):
            raise ValueError("Calibration Data invalid")
        self.ADC_BOARD.insert( { { "serien_no": serien_no, "calibration": calib} })
        
    def get_fpga(self, fpga_id):
        '''Return one FPGA given by the fpga ID

        Args:
            fpga_id: The ID of the FPGA board [0..11]
        '''

        return self.FPGA.find_one({'logicalNumber': fpga_id})
        
    def get_dnc(self, fpga_id, dnc_id):
        '''Return one DNC given by the fpga ID and dnc port.

        Args:
            fpga_id: The ID of the FPGA board [0..11]
            dnc_id: The dnc port [0..3]
        '''

        return self.DNC.find_one({'logicalNumber': dnc_id, 'parent_fpga': fpga_id})
        
    def get_dnc_id(self, dnc_id):
        '''Return one DNC given by the dnc id.

        Args:
            dnc_id: The dnc unique ID
        '''

        return self.DNC.find_one({'logicalNumber': dnc_id})

    def get_hicann(self, fpga_id, dnc_port, hicann_channel):
        '''Return one HICANN given by the fpga ID

        Args:
            fpga_id: The ID of the FPGA board [0..11]
            dnc_port: The dnc port [0..3]
            hicann_channel: The relative hicann ID
        '''

        return self.HICANN.find_one({'dncHicannChannel': hicann_channel, 'parent_fpga': fpga_id, 'parent_dnc': dnc_port})
        
    def get_hicann_id(self, hicann_id):
        '''Return one HICANN.

        Args:
            hicann_id: The hicann ID
        '''

        return self.HICANN.find_one({'uniqueId': hicann_id})
    
    def check_hicann_calibration(self, hicann_id):
        '''Check if an HICANN chip is calibrated for the LIF model parameters

        Args:
            hicann_id: The id of the desired HICANN
        '''
        
        hicann = self.HICANN.find_one({'uniqueId' : hicann_id})
        neurons = hicann['neurons']
        
        parameters = ["EL","Vreset","Vt","gL","tauref"] # Which parameters to check
        hicann_calibrated = False
        
        # Check parameters
        for p in parameters:
            for n,neuron in enumerate(neurons):
                if neuron[p + "_calibrated"]:
                    hicann_calibrated = True
                else:
                    hicann_calibrated = False
                    break
                    
        if hicann_calibrated: # Update HICANN status
            self.change_parameter_hicann(h,'calibrated',True)

    ## Return one neuron object n from HICANN h
    # @param h The unique ID of the desired HICANN
    # @param n The number of the disired neuron
    def get_neuron(self,h,n):

        # Get hicann h
        hicann = self.HICANN.find_one({'uniqueId' : h})
        
        return hicann['neurons'][n]

    ## Return all neurons from HICANN h
    # @param h The unique ID of the desired HICANN
    def get_neurons(self,h):

        # Get neuron n from hicann h
        return self.HICANN.find_one({'uniqueId' : h})['neurons']

    ## Fill HICANN h with dummy data
    # @param h The unique ID of the desired HICANN
    def fill_dummy(self,h):

        # Get neurons from hicann h
        neurons = self.get_neurons(h)
            
        # For each neuron, fill with dummy data
        for n, neuron in enumerate(neurons):
            for p in self.parameters:
                neurons[n][p] = [range(5),range(5)] 
                neuron[p + '_dev'] = range(5)
                neuron[p + '_fit'] = range(3)

        # Update HICANN
        self.HICANN.update({'uniqueId' : h}, {'$set':{'neurons' : neurons}})
        
    ## Change one parameter of a neuron
    # @param h The unique ID of the desired HICANN
    # @param n The number of the desired neuron
    # @param param The parameter to change
    # @param value The new value for the parameter
    def change_parameter_neuron(self,h,n,param,value):

        # Get hicann h neurons
        hicann = self.HICANN.find_one({'uniqueId' : h})
        neurons = hicann['neurons']

        # Modify the data
        neurons[n][param] = value

        # Update HICANN
        self.HICANN.update({'uniqueId' : h}, {'$set':{'neurons' : neurons}})
        
    ## Change one parameter of a hicann
    # @param h The unique ID of the desired HICANN
    # @param param The parameter to change
    # @param value The new value for the parameter
    def change_parameter_hicann(self,h,param,value):

        # Update HICANN
        self.HICANN.update({'uniqueId' : h}, {'$set':{param : value}})
    
    ## Change the analog readout of one HICANN
    # @param h The unique ID of the desired HICANN
    # @param param The parameter to change
    # @param value The new value for the parameter
    def change_analog_readout_hicann(self,h, adc0, channel_adc0, adc1, channel_acd1):
        analogReadout = [
                { "adc" : adc0, "channel" : channel_adc0 },
                { "adc" : adc1, "channel" : channel_adc1 } ]
        # Update HICANN
        self.HICANN.update({'uniqueId' : h}, {'$set':{ 'analogReadout' : analogReadout } } )


    ## Reset calibration status for one parameter
    # @param h The unique ID of the desired HICANN
    # @param param The parameter to reset
    def reset_status(self,h,param,neuron_index=[]):

        # Get hicann h neurons
        hicann = self.HICANN.find_one({'uniqueId' : h})
        neurons = hicann['neurons']
        
        # Reset calibration status
        if (neuron_index == []):
            neuron_index = neurons
                
        for n, neuron in enumerate(neuron_index):
            
            # Modify the data
            neurons[n][param + '_calibrated'] = False

        # Reinsert data
        self.HICANN.update({'uniqueId' : h}, {'$set':{'neurons' : neurons}})
        
    ## Reset activation status for one chip
    # @param h The unique ID of the desired HICANN
    def reset_activation_status(self,h):

        # Get hicann h neurons
        hicann = self.HICANN.find_one({'uniqueId' : h})
        neurons = hicann['neurons']

        for n, neuron in enumerate(neurons):
        
            # Modify the data
            neurons[n]['available'] = True

        # Reinsert data
        self.HICANN.update({'uniqueId' : h}, {'$set':{'neurons' : neurons}})

    ## Deactivate a neuron
    # @param h The unique ID of the desired HICANN
    # @param n The number of the desired neuron
    def deactivate_neuron(self,h,n):
        self.change_parameter_neuron(h,n,'available',False)
        
    ## Activate neuron
    # @param h The unique ID of the desired HICANN
    # @param n The number of the desired neuron
    def activate_neuron(self,h,n):
        self.change_parameter_neuron(h,n,'available',True)

    ## Activate FPGA
    # @param f The unique ID of the FPGA to be activated
    def activate_fpga(self,f):
        self.FPGA.update({'uniqueId' : f}, {'$set':{'online' : True}})
    
    ## Activate DNC
    # @param d The DNC to be activated
    # @param f The parent FPGA
    def activate_dnc(self,d,f):
        self.DNC.update({'fpgaDncChannel' : d, 'parent_fpga' : f}, {'$set':{'available' : True}})

    ## Deactivate DNC
    # @param d The DNC to be activated
    # @param f The parent FPGA
    def deactivate_dnc(self,d,f):
        self.DNC.update({'fpgaDncChannel' : d, 'parent_fpga' : f}, {'$set':{'available' : False}})

    ## Deactivate FPGA
    # @param f The unique ID of the FPGA to be activated
    def deactivate_fpga(self,f):
        self.FPGA.update({'uniqueId' : f}, {'$set':{'online' : False}})

    ## Activate HICANN
    # @param h The HICANN to be activated
    # @param d The parent DNC
    # @param f The parent FPGA
    def activate_hicann(self,h,d,f):
        self.HICANN.update({'dncHicannChannel' : h, 'parent_dnc' : d, 'parent_fpga' : f}, {'$set':{'available' : True}})
   
    ## Deactivate HICANN
    # @param h The HICANN to be activated
    # @param d The parent DNC
    # @param f The parent FPGA
    def deactivate_hicann(self,h,d,f):
        self.HICANN.update({'dncHicannChannel' : h, 'parent_dnc' : d, 'parent_fpga' : f}, {'$set':{'available' : False}})

    ## Translate a biological scaled parameter to his hardware counterpart   
    # @param h The desired HICANN
    # @param n The desired neuron
    # @param param The parameter in the scaled domain
    # @param value The value of the parameter in the scaled domain
    def bio_to_hw(self,h,n,param,value): 
        currentNeuron = self.get_neuron(h,n)

        S = param + '_fit'
        fit = currentNeuron[S]      

        a = fit[0]
        b = fit[1]
        c = fit[2]
      
        return a*value*value + b*value + c
        
    ## Translate back a hardware parameter to his scaled counterpart   
    # @param h The desired HICANN
    # @param n The desired neuron
    # @param param The parameter in the scaled domain
    # @param value The value of the parameter in the scaled domain
    def hw_to_scaled(self,h,n,param,value): 
        currentNeuron = self.get_neuron(h,n)

        fit = currentNeuron[param + '_fit']      

        a = fit[0]
        b = fit[1]
        c = fit[2]
        
        # Calculate reverse function
        delta = b*b - 4*a*(c-value)
        
        return (-b + numpy.sqrt(delta))/(2*a)
    
    ## Get the translation factors for one parameter
    # @param h The desired HICANN
    # @param n The desired neuron
    # @param param The parameter to get the translation factors from
    def get_fit(self,h,n,param):
        currentNeuron = self.get_neuron(h,n)
        fit = currentNeuron[param + '_fit']
        return fit
        
    ## List all defect neurons on one chip
    # @param h The desired HICANN
    def list_defect_neurons(self,h):
        neurons = self.get_neurons(h)
        for k in neurons:
            if (k['available'] == False):
                print k['logicalNumber']
                
    ## List all non-calibrated neuron on a chip for a given parameter
    # @param h The desired HICANN
    # @param param The parameter to check
    def list_non_calibrated(self,h,param):
        neurons = self.get_neurons(h)
        for k in neurons:
            if (k[param + '_calibrated'] == False):
                print k['logicalNumber']
        
    ## Get the mean translation factor
    # @param h The desired HICANN
    # @param start The first neuron
    # @param stop The stop neuron
    # @param param The desired parameter
    def get_mean_factor(self,h,start,stop,param):
        
        nb_points = len(self.get_neuron(h,start)[param][0])
        fg_values = self.get_neuron(h,start)[param][1]
        eff_values_mean = []
        eff_values_err = []
        for i in range(nb_points):
            temp_array = []
            for k in numpy.arange(start,stop):
                currentNeuron = self.get_neuron(h,k)
                arrayParam = currentNeuron[param]

                temp_array.append(arrayParam[0][i])
                
            eff_values_mean.append(numpy.mean(temp_array))
            eff_values_err.append(numpy.std(temp_array))
        
        # Calculate mean translation factor
        a,b,c = numpy.polyfit(eff_values_mean,fg_values,2)
        
        fg_valuesC = []
        for i in eff_values_mean:
            fg_valuesC.append(a*i*i+b*i+c)
            
        return a,b,c
    
    ## Evaluate the database to mark bad neurons    
    # @param hicann The desired HICANN
    # @param range The number of neurons to be tested
    # @param p The parameter to be tested
    def evaluate_db(self,hicann,range,p):
        neurons = self.get_neurons(hicann)
        # Go through all neurons
        for k in range:
                            
            # Get mean and std
            try:
                mean = neurons[k][p][0]
                std = neurons[k][p + '_dev']
                
                # Check for bad neurons
                for v,item in enumerate(mean):
                    if (item == 0):
                        self.deactivateNeuron(hicann,k)
                    else:
                        if (std[v]/item > 0.05):
                            self.deactivateNeuron(hicann,k)
            except:
                # If no calibration data available, just pass
                print 'passed'
                pass
                            
    # Erase DB
    def clear_db(self):
        for c in self.collections:
            c.remove()

    # Is database empty ?
    def is_empty(self):
        
        i =0
        for item in self.HICANN.find():
            i = i + 1
        
        if (i == 0):
            return True
        else:
            return False    

    ## Create neuron index for calibration
    # @param param The parameter to be calibrated
    # @param neurons_limit Limit the number of neurons to be calibrated
    # @param fpga The id of the current fpga board
    def create_neuron_index(self,param,neurons_range=[],fpga=0):
        
        # For each available HICANN, return list of neurons that are not calibrated
        hicann_index = []
        hicanns = []
        neuron_index = []

        # Create index        
        for h in self.HICANN.find({'available' : True}).sort('uniqueId', pymongo.ASCENDING):
                        
            if (h['parent_fpga'] == fpga):
                hicann_index.append(h['uniqueId'])
                hicanns.append(h)
                                
        for h in hicanns:
        
            current_hicann = []
            
            for n in h['neurons']:
                
                # Check if the neuron is calibrated and superior to 0
                if (n[param + '_calibrated'] == False and n['logicalNumber'] >= 0 and n['available'] == True):

                    # Check if neuron limit is present
                    if (neurons_range == []):
                        current_hicann.append(n['logicalNumber'])

                    if (neurons_range != [] and (n['logicalNumber'] in neurons_range)):
                        current_hicann.append(n['logicalNumber'])

            neuron_index.append(current_hicann)
            
        return hicann_index,neuron_index
    
    ## List all elements in the database
    def list_db_all(self):
        for c in self.collections:
            i =0
            for item in c.find():
                #print item
                #print ''
                i = i + 1

            print i, ' elements in the collection ', c

    ## Export all parameters of the database to JSON files
    # @param collections The collections to export. Option are "all", or "name_of_collection"
    # @param path The path to export the JSON
    # @param option The option to export files. "calibration" will export all the database, whereas "mapping" will only export mapping-related data.
    def export_json(self,collections,path,option='calibration'):
        
        # Export DB
        if (collections == 'all'):
            for c in ['WAFER','FPGA','DNC','HICANN']:
                os.system('mongoexport -c ' + c + ' -d calibrationDB -o ' + path + c + '.json')
                
                # Post process
                if (c == 'WAFER'):
                    WS_array = []
                    for item in self.WAFER.find():
                        del item['_id']
                        WS_array.append(item)
                        
                    # Prepare JSON output
                    WS_export = {'wafer' : WS_array}
                    json_export = json.dumps(WS_export)
                                
                    # Write to file
                    f = open(path + c + '.json', 'w')
                    f.write(str(json_export))
                    f.close()
                    
                if (c == 'FPGA'):
                    FPGA_array = []
                    for item in self.FPGA.find():
                        del item['_id']
                        FPGA_array.append(item)
                        
                    # Prepare JSON output
                    FPGA_export = {'fpga' : FPGA_array}
                    json_export = json.dumps(FPGA_export)
                                
                    # Write to file
                    f = open(path + c + '.json', 'w')
                    f.write(str(json_export))
                    f.close()
                    
                if (c == 'DNC'):
                    DNC_array = []
                    for item in self.DNC.find():
                        del item['_id']
                        DNC_array.append(item)
                        
                    # Prepare JSON output
                    DNC_export = {'dnc' : DNC_array}
                    json_export = json.dumps(DNC_export)
                                
                    # Write to file
                    f = open(path + c + '.json', 'w')
                    f.write(str(json_export))
                    f.close()
                    
                if (c == 'HICANN'):
                    if (option=='calibration'):
                        HICANN_array = []
                        for item in self.HICANN.find():
                            del item['_id']
                            HICANN_array.append(item)
                            
                    if (option=='mapping'):
                        HICANN_array = []
                        for item in self.HICANN.find():
                            del item['_id']
                            
                            # Only export defect neurons
                            defect_neurons = []
                            for n in item['neurons']:
                                if (n['available'] == False):
                                    defect_neuron_id = n['logicalNumber']
                                    neuron = {'logicalNumber' : defect_neuron_id, 'available' : False}
                                    defect_neurons.append(neuron)
                            item['neurons'] = defect_neurons
                            HICANN_array.append(item)
                        
                    # Prepare JSON output
                    HICANN_export = {'hicann' : HICANN_array}
                    json_export = json.dumps(HICANN_export)
                                
                    # Write to file
                    f = open(path + c + '.json', 'w')
                    f.write(str(json_export))
                    f.close()
                    
                # Export system JSON
                system = {'name' : 'brainscales-system','toplevel' : 'wafer'}
                json_export = json.dumps(system)
                
                # Write to file
                f = open(path + 'SYSTEM' + '.json', 'w')
                f.write(str(json_export))
                f.close()

    ## Import JSON files into the DB and overwrite it
    # @param path Path of the JSON files
    def import_json(self,path):
        
        # Delete existing DB
        self.clear_db()

        # Import JSON to DB
        for c in ['WAFER','FPGA','DNC','HICANN']:
                os.system('mongoimport -c ' + c + ' -d calibrationDB ' + path + c + '.json')

    ## Return the location of a given FPGA
    # @param f The FPGA id
    def get_location(self,f):

        locations = ['S','E','N','W','SSW','SSE','ESE','ENE','NNE','NNW','WNM','WSW']

        return locations[f]

    ## Return the reticleID given the FPGA and DNC numbers
    # @param f The FPGA id
    # @param d the DNC id
    def get_reticle_id(self,f,d):

        return int(self.reticle_map[f][d][0]['R'])

    ## Return the reticle X position given the FPGA, DNC and dncHicann channel
    # @param f The FPGA id
    # @param d The DNC id
    # @param c The dnc-hicann channel
    def get_reticle_xpos(self,f,d,c):

        return int(self.reticle_map[f][d][c]['RX'])

    ## Return the reticle Y position given the FPGA, DNC and dncHicann channel
    # @param f The FPGA id
    # @param d The DNC id
    # @param c The dnc-hicann channel
    def get_reticle_ypos(self,f,d,c):

        return int(self.reticle_map[f][d][c]['RY'])

    ## Return the HICANN X position given the FPGA, DNC and dncHicann channel
    # @param f The FPGA id
    # @param d The DNC id
    # @param c The dnc-hicann channel
    def get_hicann_xpos(self,f,d,c):

        return int(self.reticle_map[f][d][c]['HX'])

    ## Return the HICANN Y position given the FPGA, DNC and dncHicann channel
    # @param f The FPGA id
    # @param d The DNC id
    # @param c The dnc-hicann channel
    def get_hicann_ypos(self,f,d,c):

        return int(self.reticle_map[f][d][c]['HY'])
        
    ## Return the DNC X position given the FPGA and DNC
    # @param f The FPGA id
    # @param d The DNC id
    def get_dnc_xpos(self,f,d):

        return int(self.reticle_map[f][d][0]['DX'])

    ## Return the DNC Y position given the FPGA and DNC
    # @param f The FPGA id
    # @param d The DNC id
    def get_dnc_ypos(self,f,d):

        return int(self.reticle_map[f][d][0]['DY'])
        
    ## Return the FPGA X position given the FPGA Id
    # @param f The FPGA id
    def get_fpga_xpos(self,f):

        return int(self.fpga_map[f][2])

    ## Return the FPGA Y position given the FPGA Id
    # @param f The FPGA id
    def get_fpga_ypos(self,f):

        return int(self.fpga_map[f][3])
        
    ## Return the HICANN confId given the FPGA, DNC and dncHicann channel
    # @param f The FPGA id
    # @param d The DNC id
    # @param c The dnc-hicann channel
    def get_conf_id(self,f,d,c):

        return int(self.reticle_map[f][d][c]['H'])

    ## Randomize information for repeaters on a given HICANN
    # @param h The hicann id
    # @param error_rate The probability to mark a given repeater as not available
    def rand_repeaters(self,f,d,h,error_rate):

        # Get HICANN h
        hicann = self.get_hicann(f,d,h)

        # Change status
        repeaters = hicann['repeaters']
        for index,block in enumerate(repeaters):
            for r in range(len(block)):
                if (numpy.random.uniform() < error_rate):
                    repeaters[index][r]['available'] = False

        # Update in DB
        self.HICANN.update({'uniqueId' : h}, {'$set':{'repeaters' : repeaters}})

    ## Randomize synapse drivers for a given HICANN
    # @param h The hicann id
    # @param error_rate The probability to mark a given repeater as not available
    def rand_syn_drivers(self,f,d,h,error_rate):

        # Get HICANN h
        hicann = self.get_hicann(f,d,h)

        # Change status
        synapse_drivers = hicann['synapse_drivers']
        for i in range(2):
            for s in range(len(synapse_drivers[0])):
                if (numpy.random.uniform() < error_rate):
                    synapse_drivers[i][s]['available'] = False

        # Update in DB
        self.HICANN.update({'uniqueId' : h}, {'$set':{'synapse_drivers' : synapse_drivers}})

    ## Randomize synapses
    # @param h The hicann id
    # @param error_rate The probability to mark a given repeater as not available
    def rand_synapses(self,f,d,h,error_rate):

        # Get HICANN h
        hicann = self.get_hicann(f,d,h)

        # Change status
        synapses = hicann['synapses']
        for i in range(2):
            for l in range(len(synapses[0])):
                for c in range(len(synapses[0][0])):
                    if (numpy.random.uniform() < error_rate):
                        synapses[i][l][c]['available'] = False

        # Update in DB
        self.HICANN.update({'uniqueId' : h}, {'$set':{'synapses' : synapses}})

