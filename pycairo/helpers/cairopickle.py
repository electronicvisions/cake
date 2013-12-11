import bz2
import pickle
import numpy as np
import pyhalbe
from sthal import StHALContainer
import os
import shutil
import matplotlib.pyplot as plt
import re

valid_params = [pyhalbe.HICANN.neuron_parameter.E_synx,
                pyhalbe.HICANN.neuron_parameter.I_spikeamp,
                pyhalbe.HICANN.neuron_parameter.V_synx,
                pyhalbe.HICANN.neuron_parameter.E_syni,
                pyhalbe.HICANN.neuron_parameter.V_syni,
                pyhalbe.HICANN.neuron_parameter.E_l,
                pyhalbe.HICANN.neuron_parameter.V_t,
                pyhalbe.HICANN.neuron_parameter.I_radapt,
                pyhalbe.HICANN.neuron_parameter.I_convi,
                pyhalbe.HICANN.neuron_parameter.I_gl,
                pyhalbe.HICANN.neuron_parameter.I_convx,
                pyhalbe.HICANN.neuron_parameter.I_gladapt,
                pyhalbe.HICANN.neuron_parameter.V_exp,
                pyhalbe.HICANN.neuron_parameter.V_syntci,
                pyhalbe.HICANN.neuron_parameter.I_intbbi,
                pyhalbe.HICANN.neuron_parameter.I_fire,
                pyhalbe.HICANN.neuron_parameter.V_syntcx,
                pyhalbe.HICANN.neuron_parameter.I_intbbx,
                pyhalbe.HICANN.neuron_parameter.I_rexp,
                pyhalbe.HICANN.neuron_parameter.I_pl,
                pyhalbe.HICANN.neuron_parameter.I_bexp]

valid_shared_params = [pyhalbe.HICANN.shared_parameter.V_clra,
                       pyhalbe.HICANN.shared_parameter.V_clrc,
                       pyhalbe.HICANN.shared_parameter.V_reset,
                       pyhalbe.HICANN.shared_parameter.V_dllres,
                       pyhalbe.HICANN.shared_parameter.V_bout,
                       pyhalbe.HICANN.shared_parameter.V_dtc,
                       pyhalbe.HICANN.shared_parameter.V_thigh,
                       pyhalbe.HICANN.shared_parameter.V_br,
                       pyhalbe.HICANN.shared_parameter.I_breset,
                       pyhalbe.HICANN.shared_parameter.V_m,
                       pyhalbe.HICANN.shared_parameter.V_dep,
                       pyhalbe.HICANN.shared_parameter.V_tlow,
                       pyhalbe.HICANN.shared_parameter.V_gmax1,
                       pyhalbe.HICANN.shared_parameter.V_gmax0,
                       pyhalbe.HICANN.shared_parameter.V_gmax3,
                       pyhalbe.HICANN.shared_parameter.V_gmax2,
                       pyhalbe.HICANN.shared_parameter.V_ccas,
                       pyhalbe.HICANN.shared_parameter.V_stdf,
                       pyhalbe.HICANN.shared_parameter.V_fac,
                       pyhalbe.HICANN.shared_parameter.V_bexp,
                       pyhalbe.HICANN.shared_parameter.I_bstim,
                       pyhalbe.HICANN.shared_parameter.V_bstdf]


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
        i=0
        expdirs = []
        for dr in dirs:
            if os.path.isfile(os.path.join(self.workdir,dr,"description.txt")):
                expdirs.append(dr)
                if prnt:
                    print "{0:02d}  {1}: ".format(i,dr), open('{}/{}/description.txt'.format(self.workdir,dr)).readline()
                    i = i+1
        return expdirs
        #return [dirs,[open('{}/{}/description.txt'.format(self.workdir,d)).readline() for d in dirs]]
        #return {d: open('{}/{}/description.txt'.format(self.workdir,d)).read() for d in dirs}
    
    
    def load_experiment(self, expname):
            """ 
            """
            if type(expname) is int:
                expname = self.list_experiments(prnt = False)[expname]
            return Cairo_experiment(os.path.join(self.workdir,expname)) 
    
    
    def delete_experiment(self, expname):
        """ Delete experiment with certain name or number.
            CAUTION: Deletes all files of an experiment!

            Args:
                expname = name or number of experiment
                
        """
        if type(expname) is int:
                expname = self.list_experiments(prnt = False)[expname]
        shutil.rmtree(os.path.join(self.workdir,expname))
    
    def get_description(self,expname):
        """ Prints whole description of one experiment
            
            Args:
                expname = name or number of experiment
        """
        if type(expname) is int:
            expname = self.list_experiments(prnt = False)[expname]
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
                expname = self.list_experiments(prnt = False)[expname]
        if append:
            f = open('{}/{}/description.txt'.format(self.workdir,expname), 'a')
        else:
            f = open('{}/{}/description.txt'.format(self.workdir,expname), 'w')
        f.write(description)

    def compare_experiments(self, parameter, experiment1, experiment2, step, repetition = None):
        """ Plot histograms that compare two experiments.
            
            Args:
                experiment1: needs to be a Cairo_experiment object
                experiment2: see experiment1
                step: int, which step do you want to compare?
                repetition: int, which repetition do you want to compare?
                            if nothing given, mean over repetitions is compared

            Returns:
                matplotlib.pyplot.figure object with the histogram
        """
        if type(experiment1) is int:
                experiment1 = self.list_experiments(prnt = False)[experiment1]
        if type(experiment2) is int:
                experiment2 = self.list_experiments(prnt = False)[experiment2]
        exp1 = self.load_experiment(experiment1)
        exp2 = self.load_experiment(experiment2)

        if type(parameter) is pyhalbe.HICANN.neuron_parameter:
            steps1 = exp1.steps
            steps2 = exp1.steps
        else:
            steps1 = exp1.shared_steps
            steps2 = exp1.shared_steps

        if repetition is None:
            data1 = exp1.mean_over_reps()[0][step]
            data2 = exp2.mean_over_reps()[0][step]
        else:
            data1 = exp1.results[step][repetition]
            data2 = exp2.results[step][repetition]

        minval = int(round(steps1[0][parameter].value * 0.9))
        maxval = int(round(steps1[len(steps1)-1][parameter].value * 1.1))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(data1, range(minval,maxval,int((maxval-minval)/70)))
        ax.hist(data2, range(minval,maxval,int((maxval-minval)/70)))
        ax.set_title("Comparison of {} values.".format(parameter.name))
        ax.set_xlabel(parameter.name)
        ax.set_ylabel("Occurences")
        return fig

    def print_errors(self, experiments):
        """ Prints errors of some experiments.

            Args:
                experiments: list of experiment names or ids

            Returns:
                None
        """
        for ex_id in experiments:
            print "Errors of experiment {}:".format(ex_id)
            if type(ex_id) is int:
                ex_id = self.list_experiments(prnt = False)[ex_id]
            exp = self.load_experiment(ex_id)
            exp.calculate_errors()

    def get_broken(self):
        """ Gives names of experiments without a results folder.

            Returns:
                List of names of folders which are not valid experiments.
                CAUTION: Does not only give experiment folders!
        """
        broken = []
        dirs = np.sort([name for name in os.listdir(self.workdir) if os.path.isdir(os.path.join(self.workdir, name))])
        for d in dirs:
            if not os.path.isdir(os.path.join(self.workdir, d, "results")):
                print "{} broken: no results folder".format(d)
                broken.append(d)
        return broken



class Cairo_experiment(object):
    def __init__(self, folder):
            """ Class that contains most data of one experiment. Traces will be loaded separately.
            """
            self.workdir = folder
            self.params = pickle.load(open('{}/parameters.p'.format(self.workdir)))
            self.steps = pickle.load(open('{}/steps.p'.format(self.workdir)))
            try:
                self.shared_params = pickle.load(open('{}/shared_parameters.p'.format(self.workdir)))
                self.shared_steps = pickle.load(open('{}/shared_steps.p'.format(self.workdir)))
            except IOError:
                print "No shared parameters/steps found"
                self.shared_steps = {}
                self.shared_params = {}
            self.results_unsorted = [pickle.load(open('{}/results/{}'.format(self.workdir,fname))) for fname in sorted(os.listdir('{}/results/'.format(self.workdir)), key=lambda x: int(re.findall("[0-9]+", x)[0]))]
            self.reps = pickle.load(open("{}/repetitions.p".format(self.workdir)))
            self.results=[]
            self.num_steps = max(len(self.steps),len(self.shared_steps))
            for step in range(self.num_steps):
                self.results.append([])
                for rep in range(self.reps):
                    try:
                        self.results[step].append(self.results_unsorted[rep + step*self.reps].values())
                    except IndexError:
                        print "Step {} Rep {} not found.".format(step,rep)
            self.results = np.array(self.results)
            try:
                self.floating_gates = [[pickle.load(open('{}/floating_gates/step{}rep{}.p'.format(self.workdir,step_id,rep_id))) for rep_id in range(self.reps)] for step_id in range(len(self.steps))] # <>
            except:
                print "No fg values from {}".format(folder)
            if os.path.isfile("{}/sthalcontainer.p".format(self.workdir)):
                self.sthalcontainer = pickle.load(open("{}/sthalcontainer.p".format(self.workdir)))
            self.stepnum = len(self.results)
    
    def get_trace(self, neuron_id, step_id = 0, rep_id = 0):
            """ Get the traces of one neurons from a specific measurement
            
                Args:
                    step_id = int
                    rep_id = int repetition
                
                Returns:
                    Numpy array of pickled traces
            """
            try:
                trace = np.array(pickle.load(open('{}/traces/step{}rep{}/neuron_{}.p'.format(self.workdir,step_id,rep_id,neuron_id))))
            except IOError:
                print 'No traces saved'
                return
            return trace
    
    def plot_trace(self, neuron_id, step_id = 0, rep_id = 0):
            """ Plot the trace of one neuron from a specific measurement
            
                Args:
                    neuron_id = int
                    step_id = int
                    rep_id = int repetition
                
                Returns:
                    pyplot figure
            """
            try:
                trace = np.array(pickle.load(open('{}/traces/step{}rep{}/neuron_{}.p'.format(self.workdir,step_id,rep_id,neuron_id))))
            except IOError:
                print 'No traces saved'
                return
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(trace[0],trace[1])
            return fig

    def mean_over_reps(self):
        """ 
            Returns: 
                mean and std (trial-to-trial variation) over all repetitions:
                [mean[step][neuron],std[step][neuron]]
        """
        return [[np.mean(self.results[step], axis = 0) for step in range(self.stepnum)],
                [np.std(self.results[step], axis = 0) for step in range(self.stepnum)]]

    
    def calculate_errors(self):
        """ Print errors of measurement.
            
            Returns:
                numpy arrays: [neuron_to_neuron,trial_to_trial]
                with array[steps][0] for mean, array[steps][1] for std
                
        """
        string = ""
        string = string + "Step\t Target \t Neuron-to-Neuron\t    Trial-to-Trial\n"
        
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

            # target:
            if len(self.steps)>0:
                target = self.steps[stepid].values()[0].value
            else:
                target = self.shared_steps[stepid].values()[0].value
            
            neuron_to_neuron.append([mean_nton,std_nton])
            trial_to_trial.append([mean_ttot,std_ttot])

            string = string + "  {0:0}\t {5:6.2f} \t ({1:6.2f} +- {2:6.2f}) \t ({3:6.2f} +- {4:6.2f})\n".format(stepid, mean_nton, std_nton, mean_ttot, std_ttot, target)
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
        xs = [step[param].value for step in self.steps.values()]
        
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
            if param in valid_params:
                neuron_params[param] = self.sthalcontainer.floating_gates.getNeuron(neuron_coord, param)

        return neuron_params


    def get_sthaldefaults_shared_parameter(self, block_id):
        """ Gives sthal default shared parameters for specific block as DAC values.

            Args: 
                block_id = int (0..3)

            Returns:
                dictionary:
                    {pyhalbe.HICANN.shared_parameter.V_reset: 300, ...}
        """
        block_coord = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(block_id))
        shared_params = {}
        for param in pyhalbe.HICANN.shared_parameter.names.values():
            if param in valid_shared_params:
                if ((param is pyhalbe.HICANN.shared_parameter.V_clrc) or (param is pyhalbe.HICANN.shared_parameter.V_bexp)) and ((block_id is 0) or (block_id is 2)):
                    continue
                elif ((param is pyhalbe.HICANN.shared_parameter.V_clra) or (param is pyhalbe.HICANN.shared_parameter.V_bout)) and ((block_id is 1) or (block_id is 3)):
                    continue
                else:
                    shared_params[param] = self.sthalcontainer.floating_gates.getShared(block_coord, param) 

        return shared_params

    def get_neuron_results(self, neuron_id, mean = True):
        """ Get the results sorted per neuron, not per step.

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

