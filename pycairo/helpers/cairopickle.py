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
            self.results = [[results_unsorted[i + stp*self.reps].values() for i in range(self.reps)] for stp in range(len(self.steps))]
            if os.path.isfile("{}/sthalcontainer.p".format(self.workdir)):
                self.sthalcontainer = pickle.load(open("{}/sthalcontainer.p".format(self.workdir)))
    
    def get_traces(self, step_id = 0, rep_id = 0):
            """ Get the traces of all neurons from a specific measurement
            
                Args:
                    step_id = int
                    rep_id = int repetition
                
                Returns:
                    Numpy array of pickled traces
            """
            traces = np.array([pickle.load(open('{}/traces/step{}rep{}/neuron_{}.p'.format(self.workdir,step_id,rep_id,nid))) for nid in range(512)])
            return traces
           
    def mean_over_reps(self):
        """ 
            Returns: 
                data that is averaged over all repetitions:
                mean_over_reps[step][neuron]
        """
        return [np.mean(self.results[step], axis = 0) for step in range(len(self.steps))]
    
    def calculate_errors(self,details = False):
        """ Print errors of measurement.
            
            Args:
                details = (True|False) give detailed list?

            Returns:
                nothing
        """
        string = ""
        string = string + "Step\t Neuron-to-Neuron\t    Trial-to-Trial\n"
        
        data = self.results

        for stepid in range(len(data[1])):
            mean_nton = np.mean(self.mean_over_reps()[stepid])
            std_nton = np.std(self.mean_over_reps()[stepid])
            mean_ttot = np.mean(np.std(data[2][stepid], axis=0))
            std_ttot = np.std(np.std(data[2][stepid], axis=0))
            
            string = string + "  {0:0}\t({1:6.2f} +- {2:6.2f}) \t ({3:6.2f} +- {4:6.2f})\n".format(stepid, mean_nton, std_nton, mean_ttot, std_ttot)
            if details:
                string = string + "Worst N-to-N neurons: {} with {} mV and {} with {} mV\n".format(np.argmin(self.mean_over_reps()[stepid]-mean_nton),np.min(self.mean_over_reps()[stepid]-mean_nton),
                                                                                     np.argmax(self.mean_over_reps()[stepid]-mean_nton),np.max(self.mean_over_reps()[stepid]-mean_nton))
                string = string + "Worst T-to-T neuron: {} with std of {}\n".format(np.argmax(np.std(data[2][stepid], axis=0)), np.max(np.std(data[2][stepid], axis=0)))
        string = string + '\n'
        print string
    
   
    def fit_data(self, param, linear_fit = False):
        """ Gives fit parameters for data.
        
            Args:
                param = string ('E_l', ...) which parameter to fit
                linear_fit = (True|False) linear fit?
        
            Returns:
                polynomial fit values
        """
        
        mean_data = self.mean_over_reps()
        mean_data_sorted = [[mean_data[sid][nid] for sid in range(len(self.steps))]for nid in range(len(self.results[0][0]))]
        if param: xs = [step[param].value for step in self.steps]
        else: xs = [step.values()[0].value for step in self.steps]
        
        fit_results = [np.polyfit(mean_data_sorted[nid], xs, 2) for nid in range(len(self.results[0][0]))]
        return fit_results



