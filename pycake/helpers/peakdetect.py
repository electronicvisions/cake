from numpy import NaN, Inf, arange, isscalar, asarray, array


def peakdet(v, delta, x=None):
    """
    Detect peaks in a vector.


    Finds the local maxima and minima ("peaks") in the vector v.

    Returns two arrays [maxtab, mintab]. Both consist of two columns.
    Column 1 contains indices in v, and column 2 the found values.

    With optional vector x the indices are replaced with the corresponding
    values in x.

    A point is considered a maximum peak if it has the maximal
    value, and was preceded (to the left) by a value lower by
    delta.

    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    """

    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    v = asarray(v)

    if len(v) != len(x):
        raise TypeError('Input vectors v and x must have same length')

    if not isscalar(delta):
        raise TypeError('Input argument delta must be a scalar')

    if delta <= 0:
        raise ValueError('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

if __name__ == "__main__":  # pragma: no cover
    """Run peakdet on example data and visualize result."""

    from matplotlib.pyplot import plot, scatter, show
    series = [0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 2, 0, 0, 0, -2, 0]
    maxtab, mintab = peakdet(series, .3)
    plot(series)
    scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='blue')
    scatter(array(mintab)[:, 0], array(mintab)[:, 1], color='red')
    show()
