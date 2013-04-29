'''Visualization code split out out the database interface'''

import numpy
import matplotlib.pyplot as plt
import pycairo.interfaces.database

def plot_parameter_multi(hicann_id, neuron_ids, param, error=True, save=False, filename='default'):
    '''Plot the relation between two parameters.

    Args:
        hicann_id: The desired HICANN
        neuron_ids: list of neurons to plot, or number of neurons to plot 
        param: The parameter to be plotted
        error: Show errorbars?
        save: Save in file?
        filename: Which name for the file?
    '''
    dbi = pycairo.interfaces.database.DatabaseInterface()
    neurons = dbi.get_neurons(hicann_id)

    if isinstance(neuron_ids, int):
        neuron_ids = range(neuron_ids)
    
    for k in neuron_ids:
        if (neurons[k]['available'] == True):
            arrayParam =  neurons[k][param]            

            S = param + '_fit'
            fit =  neurons[k][S]      

            a = fit[0]
            b = fit[1]
            c = fit[2]

            minValue = min(arrayParam[0])
            maxValue = max(arrayParam[0])
          
            resolution = 20
            step = (maxValue-minValue)/resolution
            cValue = []
            
            value = numpy.arange(minValue-step,maxValue+step,step)
         
            for i in value:
                cValue.append(a*i*i+b*i+c)

            if (k < 128):
                plt.plot(value,cValue,c='r')
            if (k > 127 and k < 256):
                plt.plot(value,cValue,c='g')
            if (k > 255 and k < 384):
                plt.plot(value,cValue,c='b')
            if (k > 383):
                plt.plot(value,cValue,c='k')
            
            # Plot deviation
            if ( neurons[k][param + '_dev'] and error==True):
                plt.errorbar(arrayParam[0],arrayParam[1],xerr= neurons[k][param + '_dev'],fmt='ro',c='k')
            
            if (not  neurons[k][param + '_dev'] or error==False):
                plt.scatter(arrayParam[0],arrayParam[1],c='k',s=30,edgecolors='none')
            
            if (save == False):
                plt.xlabel('Effective hardware value')
                plt.ylabel('Floating gate value')
            
            if (save == True):
                plt.xlabel('Effective hardware value')
                plt.ylabel('Floating gate value')
                plt.savefig(filename)
                
    plt.show()
    
def plot_parameter_all(hicann_id, start, stop, param, plot_type='plot'):
    '''Plot all calibration data for one parameter.

    Args:
        hicann_id: The desired HICANN
        start: The first neuron to be plotted
        stop: The last neuron to be plotted
        param: The parameter to be plotted
        plot_type: Choose between 'plot' and 'hist'
    '''
    dbi = pycairo.interfaces.database.DatabaseInterface()
    neurons = dbi.get_neurons(hicann_id)
    
    # Find len of measurement array
    for k in numpy.arange(start,stop):
        if (neurons[k]['available'] == True and neurons[k][param +'_calibrated'] == True):
            len_array =  len(neurons[k][param][0])
            break

    # Plot all data for param
    for i in range(len_array):
        value_array = []
        
        for k in numpy.arange(start,stop):
            if (neurons[k]['available'] == True and neurons[k][param +'_calibrated'] == True):
                arrayParam =  neurons[k][param]            
                value = arrayParam[0][i]
                value_array.append(value)
        if (plot_type == 'plot'):
            plt.plot(range(len(value_array)),value_array)
        if (plot_type == 'hist'):
            plt.hist(value_array,bins=50)
            plt.ylabel('Count')
            if (param == 'EL'):
                plt.xlabel('Resting potential [mV]')
            if (param == 'gL'):
                plt.xlabel('Leakage conductance [nS]')
                
        print 'Mean : ', numpy.mean(value_array)
        print 'Std : ', numpy.std(value_array)
        
    plt.show()
    
