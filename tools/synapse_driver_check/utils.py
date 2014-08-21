import numpy as np
from scipy.optimize import curve_fit

ADC_FREQUENCY = 96.01466e6

class ValueStorage(dict):
    def __init__(self, defaults={}):
        dict.__init__(self)
        dict.update(self, defaults)

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __setattr__(self, key, val):
        return self.__setitem__(key, val)
    
def fit(xdata, ydata, func, guess=None, yerr=None):
    params, cov = curve_fit(func, xdata, ydata, guess, sigma=yerr)

    errors = np.sqrt(cov.diagonal())

    if yerr != None:
        chisq = sum(((ydata - func(xdata, *params)) / yerr)**2)
        dof = ydata.size - guess.size

        quality = (chisq, chisq/dof)
    else:
        quality = None

    return (params, errors, quality)

def repeat_spike_train(spike_train, n, duration):
    return np.concatenate([spike_train + i*duration for i in range(n)])

def cut_trace(trace, n, duration):
    return np.reshape(trace[:n*int(duration*ADC_FREQUENCY)], (n, int(duration*ADC_FREQUENCY)))

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def find_maxima(times, trace):
    smth_trace = np.mean(rolling_window(trace, 5), -1)
    dvtn_trace = np.std(rolling_window(trace, 5), -1)

    maxima = np.ndarray((3, 0))
    for i, t in enumerate(times):
        lower = t
        if i != times.size-1:
            upper = times[i+1]
        else:
            upper = smth_trace.size - 1
        span = smth_trace[lower:upper]
        
        pos = lower + np.argmax(span) + 1
        val = smth_trace[pos]
        err = dvtn_trace[pos]
        maxima = np.concatenate([maxima, np.array([pos, val, err]).reshape((3, 1))], axis=1)

    return maxima

def find_minima(times, trace):
    smth_trace = np.mean(rolling_window(trace, 5), -1)
    dvtn_trace = np.std(rolling_window(trace, 5), -1)
    
    minima = np.ndarray((3, 0))
    for i, t in enumerate(times):
        lower = t - 5
        upper = t + 5
        span = smth_trace[lower:upper]
        
        pos = lower + np.argmin(span) + 1
        val = smth_trace[pos]
        err = dvtn_trace[pos]
        minima = np.concatenate([minima, np.array([pos, val, err]).reshape((3, 1))], axis=1)

    return minima
