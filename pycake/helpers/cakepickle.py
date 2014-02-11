import bz2
import pickle
import numpy as np
import pyhalbe
from sthal import StHALContainer
import os
import shutil
import matplotlib.pyplot as plt
import re
import sys

from pyhalbe.HICANN import neuron_parameter, shared_parameter

# hack to enable pickling of old experiments from back in the days when cake was the capitol of Egypt
import pycake
sys.modules['pycairo'] = pycake 

valid_parameters = [neuron_parameter.E_synx,
                neuron_parameter.I_spikeamp,
                neuron_parameter.V_synx,
                neuron_parameter.E_syni,
                neuron_parameter.V_syni,
                neuron_parameter.E_l,
                neuron_parameter.V_t,
                neuron_parameter.I_radapt,
                neuron_parameter.I_convi,
                neuron_parameter.I_gl,
                neuron_parameter.I_convx,
                neuron_parameter.I_gladapt,
                neuron_parameter.V_exp,
                neuron_parameter.V_syntci,
                neuron_parameter.I_intbbi,
                neuron_parameter.I_fire,
                neuron_parameter.V_syntcx,
                neuron_parameter.I_intbbx,
                neuron_parameter.I_rexp,
                neuron_parameter.I_pl,
                neuron_parameter.I_bexp]


valid_shared_parameters = [shared_parameter.V_clra,
                       shared_parameter.V_clrc,
                       shared_parameter.V_reset,
                       shared_parameter.V_dllres,
                       shared_parameter.V_bout,
                       shared_parameter.V_dtc,
                       shared_parameter.V_thigh,
                       shared_parameter.V_br,
                       shared_parameter.I_breset,
                       shared_parameter.V_m,
                       shared_parameter.V_dep,
                       shared_parameter.V_tlow,
                       shared_parameter.V_gmax1,
                       shared_parameter.V_gmax0,
                       shared_parameter.V_gmax3,
                       shared_parameter.V_gmax2,
                       shared_parameter.V_ccas,
                       shared_parameter.V_stdf,
                       shared_parameter.V_fac,
                       shared_parameter.V_bexp,
                       shared_parameter.I_bstim,
                       shared_parameter.V_bstdf]

valid_params = valid_parameters
valid_shared_params = valid_shared_parameters


class Experimentreader(object):
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
    
    def load_experiment(self, expname):
            """ 
                """
            if isinstance(expname, int):
                expname = self.list_experiments(prnt = False)[expname]
            return Experiment(os.path.join(self.workdir,expname)) 
    
    
    def delete_experiment(self, expname):
        """ Delete experiment with certain name or number.
            CAUTION: Deletes all files of an experiment!

            Args:
                expname = name or number of experiment
                
        """
        if isinstance(expname,int):
                expname = self.list_experiments(prnt = False)[expname]
        shutil.rmtree(os.path.join(self.workdir,expname))
    
    def get_description(self,expname):
        """ Prints whole description of one experiment
            
            Args:
                expname = name or number of experiment

            Returns:
                string with description of experiment
        """
        if isinstance(expname,int):
            expname = self.list_experiments(prnt = False)[expname]
        return open('{}/{}/description.txt'.format(self.workdir,expname)).read() 

    def change_description(self, expname, description, append = False):
        """ Change the description of experiment with name or number to string description.
            In list_experiments() list, only first line of description is shown!
            
            Args:
                expname = name or number of experiment
                description = string containing the new description
                append = (True|False) append to existing description. If False, description is overwritten.
        """
        if isinstance(expname,int):
                expname = self.list_experiments(prnt = False)[expname]
        if append:
            f = open('{}/{}/description.txt'.format(self.workdir,expname), 'a')
        else:
            f = open('{}/{}/description.txt'.format(self.workdir,expname), 'w')
        f.write(description)

    def compare_experiments(self, experiment1, experiment2, step, parameter = None, repetition = None):
        """ Plot histograms that compare two experiments.
            
            Args:
                experiment1: needs to be a cakepickle.experiment object
                experiment2: see experiment1
                step: int, which step do you want to compare?
                repetition: int, which repetition do you want to compare?
                            if nothing given, mean over repetitions is compared
                parameter:  for older pickles, you need to specify the parameter.
                            files pickled after 2014-02-07 do not need this as they save the target parameter

            Returns:
                matplotlib.pyplot.figure object with the histogram
        """
        if type(experiment1) is int:
                experiment1 = self.list_experiments(prnt = False)[experiment1]
        if type(experiment2) is int:
                experiment2 = self.list_experiments(prnt = False)[experiment2]
        exp1 = self.load_experiment(experiment1)
        exp2 = self.load_experiment(experiment2)

        if not (isinstance(parameter, neuron_parameter) or isinstance(parameter, shared_parameter)):
            if not (hasattr(exp1, 'target_parameter') or hasattr(exp2, 'target_parameter')):
                raise ValueError("No parameter type given")
            if (exp1.target_parameter is not exp2.target_parameter):
                raise TypeError("Both experiments must have the same target parameter.")
            else:
                parameter = exp1.target_parameter

        steps1 = exp1.steps
        steps2 = exp1.steps

        if repetition is None:
            data1 = exp1.mean_over_repetitions()[0][step]
            data2 = exp2.mean_over_repetitions()[0][step]
        else:
            data1 = exp1.results[step][repetition]
            data2 = exp2.results[step][repetition]

        # Use step of neuron 0... not pretty but working
        neuron0 = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(0))
        target = steps1[step][neuron0][parameter].value
        minval = int(round(steps1[0][neuron0][parameter].value * 0.9))
        maxval = int(round(steps1[len(steps1)-1][neuron0][parameter].value * 1.1))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        h1 = ax.hist(data1, range(minval,maxval,int((maxval-minval)/70)), label = 'uncalibrated')
        h2 = ax.hist(data2, range(minval,maxval,int((maxval-minval)/70)), label = 'calibrated')
        max_y = max(max(h1[0]),max(h2[0]))
        ax.vlines(target, 0, max_y, linestyle = 'dashed', color = 'k', label = 'target')
        ax.set_title("Comparison of {} values.".format(parameter.name))
        ax.set_xlabel(parameter.name)
        ax.set_ylabel("Occurences")
        ax.legend(loc = 0)
        return fig

    def print_results(self, experiments):
        """ Prints errors of some experiments.

            Args:
                experiments: list of experiment names or ids

            Returns:
                None
        """
        for ex_id in experiments:
            print "Errors of experiment {} \n{}".format(ex_id, self.get_description(ex_id))
            if type(ex_id) is int:
                ex_id = self.list_experiments(prnt = False)[ex_id]
            exp = self.load_experiment(ex_id)
            exp.calculate_results()

    def get_broken(self):
        """ Gives names of folders that don't have a results folder.

            Returns:
                List of names of folders which are not valid experiments.
                CAUTION: Does not only give experiment folders, so don't just delete all broken experiments!
        """
        broken = []
        dirs = np.sort([name for name in os.listdir(self.workdir) if os.path.isdir(os.path.join(self.workdir, name))])
        for d in dirs:
            if not os.path.isdir(os.path.join(self.workdir, d, "results")):
                print "{} broken: no results folder".format(d)
                broken.append(d)
        return broken



class Experiment(object):
    def __init__(self, folder):
            """ Class that contains most data of one experiment. Traces will be loaded separately.
            """
            self.workdir = folder

            results_folder = os.path.join(self.workdir, 'results')
            trace_folder = os.path.join(self.workdir, 'traces')
            fgcontrol_folder = os.path.join(self.workdir, 'floating_gates')

            picklefiles = [fname for fname in os.listdir(self.workdir) if fname[-1] == 'p']

            for pfile in picklefiles:
                fullpath = os.path.join(self.workdir, pfile)
                setattr(self, pfile[:-2], pickle.load(open(fullpath)))

            if not hasattr(self, 'repetitions'):
                self.repetitions = self.parameterfile['repetitions']

            result_files = sorted(os.listdir(results_folder), key=lambda x: int(re.findall("[0-9]+", x)[0]))

            self.results_unsorted = [pickle.load(open(os.path.join(results_folder, fname))) for fname in result_files]

            self.results=[]
            self.num_steps = len(self.steps)
            for step in range(self.num_steps):
                self.results.append([])
                for rep in range(self.repetitions):
                    try:
                        self.results[step].append(self.results_unsorted[rep + step*self.repetitions].values())
                    except IndexError:
                        print "Step {} Rep {} not found.".format(step,rep)
            self.results = np.array(self.results)

            self.floating_gates = list(list())
            for step_id in range(self.num_steps):
                self.floating_gates.append(list())
                for rep_id in range(self.repetitions):
                    self.floating_gates[step_id].append(list())
                    fgfile = os.path.join(fgcontrol_folder, 'step{}rep{}.p'.format(step_id, rep_id))
                    try:
                        self.floating_gates[step_id][rep_id] = pickle.load(open(fgfile))
                    except IOError:
                        print 'No floating gates after step {} / rep {}'.format(step_id, rep_id)
                        break
                else:
                    continue  # executed if the loop ended normally (no break)
                break  # executed if 'continue' was skipped (break)

            self.num_steps = len(self.results)

    def pickle_load(self, folder, filename):
        fullpath = os.path.join(folder, filename)
        if (os.path.exists(fullpath + ".bz2")):
            fullpath += ".bz2"
        if fullpath.endswith(".bz2"):
            with bz2.BZ2File(fullpath, 'r') as f:
                return pickle.load(f)
        else:
            with open(fullpath, 'r') as f:
                return pickle.load(f)

    def get_steps(self, parameter):
        ret = []
        for step in self.steps:
            v = step.itervalues().next()
            ret.append(v[parameter].value)
        return ret

    def _get_trace(self, neuron_id, step_id, rep_id, pattern):
        """ Implementation for get_trace and get_averaged_trace"""
        try:
            path = '{}/traces/step{}rep{}'.format(self.workdir,step_id,rep_id)
            trace = self.pickle_load(path, pattern.format(neuron_id))
        except IOError:
            print 'No traces saved'
            return
        return trace

    def get_trace(self, neuron_id, step_id = 0, rep_id = 0):
        """ Get the traces of one neurons from a specific measurement

            Args:
                step_id = int
                rep_id = int repetition

            Returns:
                Numpy array of pickled traces
        """
        return self._get_trace(neuron_id, step_id, rep_id, 'neuron_{}.p')

    def get_averaged_trace(self, neuron_id, step_id = 0, rep_id = 0):
        """ Get the traces of one neurons from a specific measurement after
        averaging of the trace (e.g. for I_gl or V_syntcx)

            Args:
                step_id = int
                rep_id = int repetition

            Returns:
                Numpy array of pickled traces
        """
        return self._get_trace(neuron_id, step_id, rep_id, 'neuron_mean_{}.p')


    def mean_over_repetitions(self):
        """ 
            Returns: 
                mean and std (trial-to-trial variation) over all repetitions:
                [mean[step][neuron],std[step][neuron]]
        """
        return [[np.mean(self.results[step], axis = 0) for step in range(self.num_steps)],
                [np.std(self.results[step], axis = 0) for step in range(self.num_steps)]]

    
    def calculate_results(self):
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

        for stepid in range(self.num_steps):
            # trial to trial:
            mean_ttot = np.mean(np.std(data[stepid], axis=0))
            std_ttot = np.std(np.std(data[stepid], axis=0))
            # neuron to neuron:
            mean_nton = np.mean(self.mean_over_repetitions()[0][stepid])
            std_nton = np.sqrt(np.std(self.mean_over_repetitions()[0][stepid])**2 - mean_ttot**2)

            # target:
            # choose neuron 0 or block 0 for reference... not pretty but working
            target = self.steps[stepid].values()[0].values()[0].value
            
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
        
        mean_data = self.mean_over_repetitions()[0]
        mean_data_sorted = [[mean_data[sid][nid] for sid in range(self.num_steps)]for nid in range(len(self.results[0][0]))]
        xs = [step[param].value for step in self.steps.values()]
        
        fit_results = [np.polyfit(mean_data_sorted[nid], xs, 2) for nid in range(len(self.results[0][0]))]
        return fit_results

    def get_sthaldefaults_neuron_parameter(self, neuron_id):
        """ Gives sthal default neuron parameters for specific neuron as DAC values.

            Args: 
                neuron_id = int (0..511)

            Returns:
                dictionary:
                    {neuron_parameter.E_l: 300, ...}
        """
        neuron_parameters = {}
        
        neuron_coord = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(neuron_id))
        for param in neuron_parameter.names.values():
            if param in valid_parameters:
                neuron_parameters[param] = self.sthalcontainer.floating_gates.getNeuron(neuron_coord, param)

        return neuron_parameters


    def get_sthaldefaults_shared_parameter(self, block_id):
        """ Gives sthal default shared parameters for specific block as DAC values.

            Args: 
                block_id = int (0..3)

            Returns:
                dictionary:
                    {shared_parameter.V_reset: 300, ...}
        """
        block_coord = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(block_id))
        shared_parameters = {}
        for param in shared_parameter.names.values():
            if param in valid_shared_parameters:
                if ((param is shared_parameter.V_clrc) or (param is shared_parameter.V_bexp)) and ((block_id is 0) or (block_id is 2)):
                    continue
                elif ((param is shared_parameter.V_clra) or (param is shared_parameter.V_bout)) and ((block_id is 1) or (block_id is 3)):
                    continue
                else:
                    shared_parameters[param] = self.sthalcontainer.floating_gates.getShared(block_coord, param) 

        return shared_parameters

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
            mean_data = self.mean_over_repetitions()
            mean_data_sorted = [[mean_data[0][sid][neuron_id] for sid in range(self.num_steps)],[mean_data[1][sid][neuron_id] for sid in range(self.num_steps)]]
            return np.array(mean_data_sorted)
        else:
            return np.array([[self.results[sid][rep][neuron_id] for sid in range(self.num_steps)] for rep in range(self.repetitions)])

    def plot_trace(self, neuron_id, step, repetition):
        """ Plot the trace of a neuron.

            Args:
                neuron_id
                step
                repetition

            Returns:
                matplotlib.pyplot.figure object
        """
        
        trace = self.get_trace(neuron_id, step, repetition)
        if trace:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(trace[0], trace[1])
            return fig
        else:
            print "No traces saved."
            return

    def plot_neuron_results(self, neuron_id, step, parameter):
        """ Plot all measurement results for one neuron.

            Args:
                neuron_id, step = int
                parameter = neuron_parameter.E_l etc.

            Returns:
                matplotlib.pyplot.figure object
        """
        xs = [step[parameter].value for step in self.steps.values()]
        ys = self.get_neuron_results(neuron_id)[0]
        y_errs = self.get_neuron_results(neuron_id)[1]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(xs,ys,y_errs)
        ax.plot(xs,xs, linestyle = "dashed", color="k", alpha = 0.8)
        return fig

    def get_broken_neurons(self):
        """ Get the indices of neurons marked as broken.

            Args:
                None

            Returns:
                List of indices, e.g. [0,1,2,3,4,5,6,7,...]
        """
        pass # TODO






class Experimentreader_old(object):
    """ Class to open experiments in a specific workfolder.
        OLD experiment reader to read experiments done before 6th of February, 2014
        
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
            if isinstance(expname,int):
                expname = self.list_experiments(prnt = False)[expname]
            return Experiment(os.path.join(self.workdir,expname)) 
    
    
    def delete_experiment(self, expname):
        """ Delete experiment with certain name or number.
            CAUTION: Deletes all files of an experiment!

            Args:
                expname = name or number of experiment
                
        """
        if isinstance(expname,int):
                expname = self.list_experiments(prnt = False)[expname]
        shutil.rmtree(os.path.join(self.workdir,expname))
    
    def get_description(self, expname):
        """ Prints whole description of one experiment
            
            Args:
                expname = name or number of experiment

            Returns:
                string with description of experiment
        """
        if isinstance(expname,int):
            expname = self.list_experiments(prnt = False)[expname]
        return open('{}/{}/description.txt'.format(self.workdir,expname)).read() 

    def change_description(self, expname, description, append = False):
        """ Change the description of experiment with name or number to string description.
            In list_experiments() list, only first line of description is shown!
            
            Args:
                expname = name or number of experiment
                description = string containing the new description
                append = (True|False) append to existing description. If False, description is overwritten.
        """
        if isinstance(expname,int):
                expname = self.list_experiments(prnt = False)[expname]
        if append:
            f = open('{}/{}/description.txt'.format(self.workdir,expname), 'a')
        else:
            f = open('{}/{}/description.txt'.format(self.workdir,expname), 'w')
        f.write(description)

    def compare_experiments(self, experiment1, experiment2, step, parameter, repetition = None):
        """ Plot histograms that compare two experiments.
            
            Args:
                experiment1: needs to be a cakepickle.experiment object
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

        if type(parameter) is neuron_parameter:
            steps1 = exp1.steps
            steps2 = exp1.steps
        else:
            steps1 = exp1.shared_steps
            steps2 = exp1.shared_steps

        if repetition is None:
            data1 = exp1.mean_over_repetitions()[0][step]
            data2 = exp2.mean_over_repetitions()[0][step]
        else:
            data1 = exp1.results[step][repetition]
            data2 = exp2.results[step][repetition]

        target = steps1[step][parameter].value
        minval = int(round(steps1[0][parameter].value * 0.9))
        maxval = int(round(steps1[len(steps1)-1][parameter].value * 1.1))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        h1 = ax.hist(data1, range(minval,maxval,int((maxval-minval)/70)), label = 'uncalibrated')
        h2 = ax.hist(data2, range(minval,maxval,int((maxval-minval)/70)), label = 'calibrated')
        max_y = max(max(h1[0]),max(h2[0]))
        ax.vlines(target, 0, max_y, linestyle = 'dashed', color = 'k', label = 'target')
        ax.set_title("Comparison of {} values.".format(parameter.name))
        ax.set_xlabel(parameter.name)
        ax.set_ylabel("Occurences")
        ax.legend(loc = 0)
        return fig

    def print_results(self, experiments):
        """ Prints errors of some experiments.

            Args:
                experiments: list of experiment names or ids

            Returns:
                None
        """
        for ex_id in experiments:
            print "Errors of experiment {} \n{}".format(ex_id, self.get_description(ex_id))
            if type(ex_id) is int:
                ex_id = self.list_experiments(prnt = False)[ex_id]
            exp = self.load_experiment(ex_id)
            exp.calculate_results()

    def get_broken(self):
        """ Gives names of folders that don't have a results folder.

            Returns:
                List of names of folders which are not valid experiments.
                CAUTION: Does not only give experiment folders, so don't just delete all broken experiments!
        """
        broken = []
        dirs = np.sort([name for name in os.listdir(self.workdir) if os.path.isdir(os.path.join(self.workdir, name))])
        for d in dirs:
            if not os.path.isdir(os.path.join(self.workdir, d, "results")):
                print "{} broken: no results folder".format(d)
                broken.append(d)
        return broken



class Experiment_old(object):
    def __init__(self, folder):
            """ Class that contains most data of one experiment. Traces will be loaded separately.
                OLD experiment to handle experiments done before 6th of February, 2014
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
            self.num_steps = len(self.results)

    def get_steps(self, parameter):
        if isinstance(parameter, neuron_parameter):
            neuron0 = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(0))
            return [j[neuron0][parameter].value for i,j in self.steps.iteritems()]
        else:
            block0 = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(0))
            return [j[block0][parameter].value for i,j in self.shared_steps.iteritems()]

    
    def get_trace(self, neuron_id, step_id = 0, rep_id = 0):
            """ Get the traces of one neurons from a specific measurement
            
                Args:
                    step_id = int
                    rep_id = int repetition
                
                Returns:
                    Numpy array of pickled traces
            """
            try:
                neuron = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(neuron_id))
                trace = np.array(pickle.load(open('{}/traces/step{}rep{}/neuron_{}.p'.format(self.workdir,step_id,rep_id,neuron))))
            except IOError:
                print 'No traces saved'
                return
            return trace


    def mean_over_reps(self):
        """ 
            Returns: 
                mean and std (trial-to-trial variation) over all repetitions:
                [mean[step][neuron],std[step][neuron]]
        """
        return [[np.mean(self.results[step], axis = 0) for step in range(self.num_steps)],
                [np.std(self.results[step], axis = 0) for step in range(self.num_steps)]]

    
    def calculate_results(self):
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
        mean_data_sorted = [[mean_data[sid][nid] for sid in range(self.num_steps)]for nid in range(len(self.results[0][0]))]
        xs = [step[param].value for step in self.steps.values()]
        
        fit_results = [np.polyfit(mean_data_sorted[nid], xs, 2) for nid in range(len(self.results[0][0]))]
        return fit_results

    def get_sthaldefaults_neuron_parameter(self, neuron_id):
        """ Gives sthal default neuron parameters for specific neuron as DAC values.

            Args: 
                neuron_id = int (0..511)

            Returns:
                dictionary:
                    {neuron_parameter.E_l: 300, ...}
        """
        neuron_params = {}
        
        neuron_coord = pyhalbe.Coordinate.NeuronOnHICANN(pyhalbe.Coordinate.Enum(neuron_id))
        for param in neuron_parameter.names.values():
            if param in valid_params:
                neuron_params[param] = self.sthalcontainer.floating_gates.getNeuron(neuron_coord, param)

        return neuron_params


    def get_sthaldefaults_shared_parameter(self, block_id):
        """ Gives sthal default shared parameters for specific block as DAC values.

            Args: 
                block_id = int (0..3)

            Returns:
                dictionary:
                    {shared_parameter.V_reset: 300, ...}
        """
        block_coord = pyhalbe.Coordinate.FGBlockOnHICANN(pyhalbe.Coordinate.Enum(block_id))
        shared_params = {}
        for param in shared_parameter.names.values():
            if param in valid_shared_params:
                if ((param is shared_parameter.V_clrc) or (param is shared_parameter.V_bexp)) and ((block_id is 0) or (block_id is 2)):
                    continue
                elif ((param is shared_parameter.V_clra) or (param is shared_parameter.V_bout)) and ((block_id is 1) or (block_id is 3)):
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
            mean_data_sorted = [[mean_data[0][sid][neuron_id] for sid in range(self.num_steps)],[mean_data[1][sid][neuron_id] for sid in range(self.num_steps)]]
            return np.array(mean_data_sorted)
        else:
            return np.array([[self.results[sid][rep][neuron_id] for sid in range(self.num_steps)] for rep in range(self.reps)])

    def plot_trace(self, neuron_id, step, repetition):
        """ Plot the trace of a neuron.

            Args:
                neuron_id
                step
                repetition

            Returns:
                matplotlib.pyplot.figure object
        """
        
        trace = self.get_trace(neuron_id, step, repetition)
        if trace:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(trace[0], trace[1])
            return fig
        else:
            print "No traces saved."
            return

    def plot_neuron_results(self, neuron_id, step, parameter):
        """ Plot all measurement results for one neuron.

            Args:
                neuron_id, step = int
                parameter = neuron_parameter.E_l etc.

            Returns:
                matplotlib.pyplot.figure object
        """
        xs = [step[parameter].value for step in self.steps.values()]
        ys = self.get_neuron_results(neuron_id)[0]
        y_errs = self.get_neuron_results(neuron_id)[1]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(xs,ys,y_errs)
        ax.plot(xs,xs, linestyle = "dashed", color="k", alpha = 0.8)
        return fig

    def get_broken_neurons(self):
        """ Get the indices of neurons marked as broken.

            Args:
                None

            Returns:
                List of indices, e.g. [0,1,2,3,4,5,6,7,...]
        """
        pass # TODO
