import numpy as np

try:
    import pyfftw.interfaces
    fft = pyfftw.interfaces.numpy_fft.rfft
    fftfreq = pyfftw.interfaces.numpy_fft.fftfreq
except ImportError as exc:  # pragma: no cover
    print exc
    print "FFT will not work!"
    print "use: pip install --upgrade --user pyFFTW"
    fft = None
    fftfreq = None


class SignalToNoise(object):
    """Utility class to determine the signal to noise ratio of the neuron membrane
    for regular spike input
    """

    def __init__(self, control, adc_freq, bg_freq, fast=True):
        """
        Args:
            control: np.array Membrane trace of the neuron without synaptic input
            adc_freq: float Sample frequencey of the ADC
            bg_freq: float Frequence of spike input
        """
        self.trace_len = len(control)
        self.n = self.trace_len
        if fast:
            self.n = 2**int(np.log2(self.trace_len))
        assert self.n <= self.trace_len

        dt = 1.0/adc_freq
        freq = fftfreq(self.n, dt)
        peak = np.searchsorted(freq[:len(freq) / 2], bg_freq)
        self.window = np.array((-2, -1, 0, 1, 2)) + peak
        spec = fft(control, n=self.n)[self.window]
        self.noise = np.sum(np.abs(spec))

    def __call__(self, trace):
        """
        Calculates the signale to noise ratio for the given trace

        Args:
            trace: np.array Membrane trace of the neuron with synaptic input

        Returns:
            float The signal to noise ratio
        """
        assert self.trace_len == len(trace)
        # Compat for older pickled traces
        n = getattr(self, 'n', self.trace_len)
        fft_trace = np.abs(fft(trace, n)[self.window])
        return np.sum(fft_trace) / self.noise
