import unittest
import numpy as np
import pycalibtic as cal
import pycairo.experiment as exp


def initBackend(fname):
    lib = cal.loadLibrary(fname)
    backend = cal.loadBackend(lib)

    if not backend:
        raise Exception("unable to load %s" % fname)

    return backend


def loadXMLBackend(path="build"):
    backend = initBackend("libcalibtic_xml.so")
    backend.config("path", path)
    backend.init()
    return backend


class FakeHICANN(object):
    """Fakes functionality that would measure on actual hardware."""
    def reset(self):
        pass

    def program_fg(self, parameters):
        pass

    def configure(self):
        pass

    def activate_neuron(self, neuron_id):
        pass

    def enable_analog_output(self, neuron_id):
        pass


class FakeADC(object):
    """Fakes ADC functionality, generates data."""
    def measure_adc(self, points):
        t = np.array([])
        v = np.array([])
        return t, v


class TestExperiment(unittest.TestCase):
    def test_Current(self):
        c = exp.Current(100)
        self.assertEqual(c.toDAC().value, 41)

    def test_Voltage(self):
        v = exp.Voltage(360)
        self.assertEqual(v.toDAC().value, 205)

    def test_DAC(self):
        d = exp.DAC(3.14)
        self.assertEqual(d.toDAC().value, 3)

    def test_Calibrate_E_l(self):
        neurons = [2, 3, 5, 7, 8, 10]
        fake_hicann = FakeHICANN()
        fake_adc = FakeADC()
        calib = exp.Calibrate_E_l(fake_hicann, fake_adc, neurons)
        self.assertTrue(False, "tests not implemented")

if __name__ == "__main__":
    unittest.main()
