import pylab
from numpy import interp


def nonint_reshape_data(data, d):
    """
    Reshape data using the floating point period d (in data samples).

    Only int(len(data) / int(d)) samples starting at index 0 are considered.
    """
    reshaped = []
    for i in xrange(int(len(data) / d)):
        offset = int(round(d * i))
        reshaped.append(data[offset: offset + int(d)])

    reshaped = pylab.array(reshaped)
    return reshaped


def nonint_reshape_data_interp(data, d):
    reshaped = []

    index_base = pylab.arange(0, int(d) + 1)

    for i in xrange(0, int(len(data) / d), 1):
        offset = int(d * i)

        if i == 0:
            new_row = interp(
                index_base[1:-1] + pylab.fmod(d * i, 1.),
                index_base[1:],
                data[offset: offset - 1 + len(index_base)]
                )
        else:
            new_row = interp(
                index_base[1:-1] + pylab.fmod(d * i, 1.),
                index_base,
                data[offset - 1: offset - 1 + len(index_base)]
                )

        reshaped.append(new_row)

    reshaped = pylab.array(reshaped)
    return reshaped


def tune_period(signal, initial_period, tune_range, steps=100,
                reshape_method=nonint_reshape_data):

    add_factors = pylab.linspace(
        tune_range[0],
        tune_range[1],
        steps)

    stds = pylab.empty(len(add_factors))

    for index, add_factor in enumerate(add_factors):
        resh = reshape_method(
            signal,
            initial_period + initial_period * add_factor)
        std = pylab.std(resh, axis=0)
        stds[index] = pylab.mean(std)

    return pylab.argmin(stds), add_factors, stds
