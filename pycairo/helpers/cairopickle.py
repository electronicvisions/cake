import bz2
import pickle
import numpy as np
import pyhalbe
from sthal import StHALContainer
import os
import shutil

class Cairo_Experimentreader(object):
    """ Class to open experiments in a specific workfolder.
        
        Args:
            folder: specify the path where experiments are stored.
                    If nothing is given, the current directory is used.
    """
    def __init__(self, folder = None):
        if folder:
            self.workdir = folder
        else:
            self.workdir = os.getcwd()

    def list_experiments(self, prnt = True):
        """ List all experiments in the folder. 

            Args:
                prnt = (True|False) do you want to print directly?

            Returns:
                if prnt = False: List of experiment names and descriptions
        """
        dirs = np.sort([name for name in os.listdir(self.workdir) if os.path.isdir(os.path.join(self.workdir, name))])
        if prnt:
            i=0
            for dr in dirs:
                print "{0:02d}  {1}: ".format(i,dr), open('{}/{}/description.txt'.format(self.workdir,dr)).readline()
                i=i+1
        if not prnt:
            return [dirs,[open('{}/{}/description.txt'.format(self.workdir,d)).readline() for d in dirs]]
        #return {d: open('{}/{}/description.txt'.format(self.workdir,d)).read() for d in dirs}
    
    
    def load_experiment(self, expname):
            """ 
            """
            if type(expname) is int:
                expname = self.list_experiments(prnt = False)[0][expname]
            return Cairo_experiment(os.path.join(self.workdir,expname)) 
    
    
    def delete_experiment(self, expname):
        """ Delete experiment with certain name or number.
            CAUTION: Deletes all files of an experiment!

            Args:
                expname = name or number of experiment
                
        """
        if type(expname) is int:
                expname = self.list_experiments(prnt = False)[0][expname]
        shutil.rmtree(os.path.join(self.workdir,expname))
    
    def get_description(self,expname):
        """ Prints whole description of one experiment
            
            Args:
                expname = name or number of experiment
        """
        if type(expname) is int:
            expname = self.list_experiments(prnt = False)[0][expname]
        print open('{}/{}/description.txt'.format(self.workdir,expname)).read() 

    def change_description(self, expname, description, append = False):
        """ Change the description of experiment with name or number to string description.
            In list_experiments() list, only first line of description is shown!
            
            Args:
                expname = name or number of experiment
                description = string containing the new description
                append = (True|False) append to existing description. If False, description is overwritten.
        """
        if type(expname) is int:
                expname = self.list_experiments(prnt = False)[0][expname]
        if append:
            f = open('{}/{}/description.txt'.format(self.workdir,expname), 'a')
        else:
            f = open('{}/{}/description.txt'.format(self.workdir,expname), 'w')
        f.write(description)
        


class Cairo_experiment(object):
    def __init__(self, folder):
            """ Class that contains most data of one experiment. Traces will be loaded separately. 
            """
            self.workdir = folder
            self.params = pickle.load(open('{}/parameters.p'.format(self.workdir)))
            self.steps = pickle.load(open('{}/steps.p'.format(self.workdir)))
            results_unsorted = [pickle.load(open('{}/results/{}'.format(self.workdir,fname))) for fname in os.listdir('{}/results/'.format(self.workdir))]
            self.reps = pickle.load(open("{}/repetitions.p".format(self.workdir)))
            self.results=[]
            for step in range(len(self.steps)):
                self.results.append([])
                for rep in range(self.reps):
                    try:
                        self.results[step].append(results_unsorted[rep + step*self.reps].values())
                    except IndexError:
                        print "Step {} Rep {} not found.".format(step,rep)
            try:
                self.floating_gates = [[pickle.load(open('{}/floating_gates/step{}rep{}.p'.format(self.workdir,step_id,rep_id))) for rep_id in range(self.reps)] for step_id in range(len(self.steps))] # <>
            except:
                print "No fg values from {}".format(folder)
            if os.path.isfile("{}/sthalcontainer.p".format(self.workdir)):
                self.sthalcontainer = pickle.load(open("{}/sthalcontainer.p".format(self.workdir)))
            self.stepnum = len(self.results)
    
    def get_traces(self, step_id = 0, rep_id = 0):
            """ Get the traces of all neurons from a specific measurement
            
                Args:
                    step_id = int
                    rep_id = int repetition
                
                Returns:
                    Numpy array of pickled traces
            """
            try:
                traces = np.array([pickle.load(open('{}/traces/step{}rep{}/neuron_{}.p'.format(self.workdir,step_id,rep_id,nid))) for nid in range(512)])
            except IOError:
                print 'No traces saved'
                pass
            return traces
           
    def mean_over_reps(self):
        """ 
            Returns: 
                data that is averaged over all repetitions:
                [mean[step][neuron],std[step][neuron]]
        """
        return [[np.mean(self.results[step], axis = 0) for step in range(self.stepnum)],
                [np.std(self.results[step], axis = 0) for step in range(self.stepnum)]]

    
    def calculate_errors(self,details = False):
        """ Print errors of measurement.
            
            Args:
                details = (True|False) give detailed list?

            Returns:
                numpy arrays: [neuron_to_neuron,trial_to_trial]
                with array[steps][0] for mean, array[steps][1] for std
                
        """
        string = ""
        string = string + "Step\t Neuron-to-Neuron\t    Trial-to-Trial\n"
        
        data = self.results

        neuron_to_neuron = []
        trial_to_trial = []

        for stepid in range(len(data)):
            # neuron to neuron:
            mean_nton = np.mean(self.mean_over_reps()[0][stepid])
            std_nton = np.std(self.mean_over_reps()[0][stepid])
            # trial to trial:
            mean_ttot = np.mean(np.std(data[stepid], axis=0))
            std_ttot = np.std(np.std(data[stepid], axis=0))
            
            neuron_to_neuron.append([mean_nton,std_nton])
            trial_to_trial.append([mean_ttot,std_ttot])

            string = string + "  {0:0}\t({1:6.2f} +- {2:6.2f}) \t ({3:6.2f} +- {4:6.2f})\n".format(stepid, mean_nton, std_nton, mean_ttot, std_ttot)
            if details:
                string = string + "Worst N-to-N neurons: {} with {} mV and {} with {} mV\n".format(np.argmin(self.mean_over_reps()[0][stepid]-mean_nton),np.min(self.mean_over_reps()[0][stepid]-mean_nton),
                                                                                     np.argmax(self.mean_over_reps()[0][stepid]-mean_nton),np.max(self.mean_over_reps()[0][stepid]-mean_nton))
                string = string + "Worst T-to-T neuron: {} with std of {}\n".format(np.argmax(np.std(data[2][stepid], axis=0)), np.max(np.std(data[2][stepid], axis=0)))
        string = string + '\n'
        print string
        return [neuron_to_neuron,trial_to_trial]
    
   
    def fit_data(self, param, linear_fit = False):
        """ Gives fit parameters for data.
        
            Args:
                param = string ('E_l', ...) which parameter to fit
                linear_fit = (True|False) linear fit?
        
            Returns:
                polynomial fit values
        """
        
        mean_data = self.mean_over_reps()[0]
        mean_data_sorted = [[mean_data[sid][nid] for sid in range(self.stepnum)]for nid in range(len(self.results[0][0]))]
        if param: xs = [step[param].value for step in self.steps]
        else: xs = [step.values()[0].value for step in self.steps]
        
        fit_results = [np.polyfit(mean_data_sorted[nid], xs, 2) for nid in range(len(self.results[0][0]))]
        return fit_results

    def get_sthaldefaults_neuron_parameter(self, neuron_id):
        """ Gives sthal default neuron parameters for specific neuron as DAC values.

            Args: 
                neuron_id = int (0..511)

            Returns:
                dictionary:
                    {pyhalbe.HICANN.neuron_parameter.E_l: 300, ...}
        """
        neuron_params = {}
        
        neuron_coord = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(neuron_id))
        for param in pyhalbe.HICANN.neuron_parameter.names.values():
            if param is not pyhalbe.HICANN.neuron_parameter.__last_neuron:
                neuron_params[param] = self.sthalcontainer.floating_gates.getNeuron(neuron_coord, i)

        return neuron_params


    def get_sthaldefaults_shared_parameter(self, block_id):
        """ Gives sthal default shared parameters for specific block as DAC values.

            Args: 
                block_id = int (0..3)

            Returns:
                dictionary:
                    {pyhalbe.HICANN.shared_parameter.V_reset: 300, ...}
        """
        block_coord = pyhalbe._Coordinate.FGBlockOnHICANN(pyhalbe._Coordinate.Enum(block_id))
        shared_params = {}
        for param in pyhalbe._HICANN.shared_parameter.names.values():
            if (i is not pyhalbe._HICANN.shared_parameter.__last_shared) and param is not (pyhalbe._HICANN.shared_parameter.int_op_bias):
                if ((param is pyhalbe._HICANN.shared_parameter.V_clrc) or (param is pyhalbe._HICANN.shared_parameter.V_bexp)) and ((block_id is 0) or (block_id is 2)):
                    continue
                elif ((param is pyhalbe._HICANN.shared_parameter.V_clra) or (param is pyhalbe._HICANN.shared_parameter.V_bout)) and ((block_id is 1) or (block_id is 3)):
                    continue
                else:
                    shared_params[param] = self.sthalcontainer.floating_gates.getShared(block_coord, param) 

    def get_neuron_results(self, neuron_id, mean = True):
        """ Get the results sorted for one neuron.

            Args:
                neuron_id = (0..511)
                mean = (True|False) mean over all repetitions?

            Returns:
                If mean = True:
                    numpy array with mean results for each neuron in first entry and (trial-to-trial) stds in second entry.
                    Like this: get_neuron_results()[mean/std][step]

                If mean = False:
                    List of arrays with each entry representing one repetition.
                    Like this: get_neuron_results()[rep][step]
        """
        if mean:
            mean_data = self.mean_over_reps()
            mean_data_sorted = [[mean_data[0][sid][neuron_id] for sid in range(self.stepnum)],[mean_data[1][sid][neuron_id] for sid in range(self.stepnum)]]
            return mean_data_sorted
        else:
            return [[self.results[sid][rep][neuron_id] for sid in range(self.stepnum)] for rep in range(self.reps)]
