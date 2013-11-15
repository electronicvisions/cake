import numpy as np
import scipy.fftpack as fft


def spectrum(trace, min_period, max_period=None):
    if not max_period:
        max_period = trace.size
    assert min_period < max_period <= trace.size

    freqs = fft.fftfreq(trace.size)
    r = np.logical_and(1.0 / max_period < freqs, freqs < 1.0 / min_period)
    freqs = freqs[r]
    f = np.abs(fft.fft(trace)[r])

    return np.vstack((freqs, f))


def find_dominant_freqs(spect, percentile=90, axis=None):
    freqs, all_peaks = spect

    peaks = all_peaks > np.percentile(all_peaks, percentile)

    if axis:
        axis.plot(freqs, all_peaks)
        for peak in freqs[peaks]:
            axis.axvline(peak)

    return np.vstack((freqs[peaks], all_peaks[peaks]))


def classify(spect, isi, logfn=None):
    freqs, peaks = spect

    # Use lambdas to defer calculations (short-circuiting).
    rules = [(
        lambda: abs(freqs[0] - 1.0 / isi) < 0.5e-3,
        lambda:
        "First Frequency {} does not match 1/(isi {}).".format(
            freqs[0], isi)
    ), (
        lambda: (lambda xs: np.allclose(xs, np.round(xs)))(freqs / freqs[0]),
        lambda:
        "Frequencies have to be multiples of first frequency: {}".format(
            freqs / freqs[0])
    ), (
        lambda: np.diff(peaks).mean() <= 0,
        lambda:
        "Mean difference of peak height has to be negative: {} {}".format(
            np.diff(peaks).mean(), peaks)
    )]

    for rule, msg in rules:
        if not rule():
            if callable(logfn):
                logfn(msg())
            return False
    return True
