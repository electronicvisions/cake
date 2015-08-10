import pycalibtic
import pywrapstdvector
import pylogging
import os
import getpass

import Coordinate
from pyhalbe.HICANN import neuron_parameter, shared_parameter
from pycake.helpers.units import DAC, Volt, Ampere, Unit, Second

def create_pycalibtic_transformation(coefficients, domain=None, trafo_type=pycalibtic.Polynomial):
    """Create a pycalibtic.Polynomial from a list of coefficients.

    Order: [c0, c1, c2, ...] resulting in c0*x^0 + c1*x^1 + c2*x^2 + ...

    Domain should be a tuple wit min and max possible hardware value"""
    # Make standard python list to have the right order
    if coefficients is None:
        return None
    coefficients = list(coefficients)
    data = pywrapstdvector.Vector_Double(coefficients)
    if domain is None:
        return trafo_type(data)
    return trafo_type(data, domain[0], domain[1])

class Calibtic(object):
    logger = pylogging.get("pycake.calibtic")

    def __setstate__(self, dic):
        # For old pickles: just set bigcap and speedup to default values
        # Otherwise, unpickling of old calibtic helpers does not work
        if 'pll' not in dic:
            dic['pll'] = 0.0
        if 'ideal_nc' not in dic:
            dic['ideal_nc'] = pycalibtic.NeuronCollection()
            dic['ideal_nc'].setDefaults()
        if 'ideal_bc' not in dic:
            dic['ideal_bc'] = pycalibtic.BlockCollection()
            dic['ideal_bc'].setDefaults()
        if not (dic.has_key('bigcap') and dic.has_key('speedup')):
            dic['bigcap'] = True
            dic['speedup'] = 'normal'
            dic['hc'] = None
        if 'hc' not in dic:
            self.hc = self.load_calibration()

    def __init__(self, path, wafer, hicann, pll=100e6, bigcap=True, speedup='normal'):
        self.path = self.make_path(path)
        self.wafer = wafer
        self.hicann = hicann
        self.pll = pll
        self.bigcap = bigcap
        self.speedup = speedup

        self.ideal_nc = pycalibtic.NeuronCollection()
        self.ideal_nc.setDefaults()
        self.ideal_bc = pycalibtic.BlockCollection()
        self.ideal_bc.setDefaults()
        self.hc = self.load_calibration()

    def get_backend(self, type='xml'):
        if type == 'xml':
            lib = pycalibtic.loadLibrary('libcalibtic_xml.so')
            backend = pycalibtic.loadBackend(lib)
            backend.config('path', self.path)
            backend.init()
            return backend
        else:
            raise ValueError("unknown backend type")

    @staticmethod
    def make_path(path):
        path = os.path.expanduser(path)
        if not os.path.isdir(path):
            Calibtic.logger.INFO("Creating backend path {}".format(path))
            os.makedirs(path)
        return path

    def clear_calibration(self):
        """ Clears all calibration data.
        """
        name = self.get_calibtic_name()
        fullname = name + ".xml"
        fullpath = os.path.join(self.path, fullname)
        if os.path.isfile(fullpath):
            msg = "Clearing calibration data by removing file {}"
            self.logger.INFO(msg.format(fullpath))
            os.remove(fullpath)

    def clear_one_calibration(self, parameter):
        """ Only clears the calibration for one parameter.
        """
        if isinstance(parameter, shared_parameter):
            collection = self.bc
            ids = range(4)
        else:
            collection = self.nc
            ids = range(512)

        for i in ids:
            self.logger.TRACE("Resetting calibration for parameter {} for id {}".format(parameter.name, i))
            collection.at(i).reset(parameter, None)

    def get_calibtic_name(self):
        wafer_id = self.wafer.value()
        hicann_id = self.hicann.id().value()
        if self.bigcap:
            cap_prefix = 'bigcap'
        else:
            cap_prefix = 'smallcap'
        name = "w{}-h{}_{}_{}".format(int(wafer_id), int(hicann_id), cap_prefix, self.speedup)
        return name

    def load_calibration(self):
        """ Load existing calibration data from backend.
        """

        hc = pycalibtic.HICANNCollection()
        md = pycalibtic.MetaData()

        # load existing calibration
        try:
            name = self.get_calibtic_name()
            backend = self.get_backend()
            backend.load(name, md, hc)
            calibration_existed = True
        except RuntimeError, e:
            if e.message != "data set not found":
                raise

            # Delete all standard entries.
            #TODO: fix calibtic to use proper standard entries

            nc = hc.atNeuronCollection()
            for nid in range(512):
                nc.erase(nid)
            bc = hc.atBlockCollection()
            for bid in range(4):
                bc.erase(bid)
            hc.setPLLFrequency(int(self.pll))

        if hc.getPLLFrequency() != self.pll:
            msg  = "PLL stored in HICANNCollection {} MHz != {} MHz set here"
            self.logger.ERROR(
                msg.format(hc.getPLLFrequency()/1e6, self.pll/1e6))
        return hc

    @property
    def nc(self):
        """Returns a pycalibtic.NeuronCollection"""
        return self.hc.atNeuronCollection()

    @property
    def bc(self):
        """Returns a pycalibtic.BlockCollection"""
        return self.hc.atBlockCollection()

    def write_calibration(self, parameter, trafos):
        """ Writes calibration data to backend

            Args:
                parameter: which neuron or hicann parameter
                trafos: dict { coord : pycalibtic.transformation }
        """
        name = self.get_calibtic_name()

        for coord, trafo in trafos.iteritems():

            if trafo is None:
                continue

            if isinstance(parameter, shared_parameter) and isinstance(coord, Coordinate.FGBlockOnHICANN):
                collection = self.bc
                cal = pycalibtic.SharedCalibration()
            else:
                collection = self.nc
                cal = pycalibtic.NeuronCalibration()

            # Readout shifts are stored as parameter Neuroncalibration.ReadoutShift
            if parameter is 'readout_shift':
                param_id = pycalibtic.NeuronCalibration.ReadoutShift
                param_name = parameter
            else:
                param_id = parameter
                param_name = parameter.name

            index = coord.id().value()

            if not collection.exists(index):
                collection.insert(index, cal)
            collection.at(index).reset(param_id, trafo)
            self.logger.TRACE("Resetting coordinate {} parameter {} to {}".format(coord, param_name, trafo))
        md = pycalibtic.MetaData()
        backend = self.get_backend()
        backend.store(name, md, self.hc)

    def get_calibration(self, coord, use_ideal=False):
        """ Returns NeuronCalibration or SharedCalibration object for one coordinate.

            If a collection is not found und use_ideal is set to True, returns an ideal calibration
            This is turned off by default.
            If no calibration is found for a coordinate, it returns an empty calibration.
        """
        c_id = coord.id().value()

        if isinstance(coord, Coordinate.FGBlockOnHICANN):
            collection = self.bc
            calibration = pycalibtic.SharedCalibration
        else:
            collection = self.nc
            calibration = pycalibtic.NeuronCalibration

        if collection.exists(c_id) and not use_ideal:
            self.logger.TRACE("Found Calibration for {}".format(coord))
            return collection.at(c_id)
        elif use_ideal:
            self.logger.WARN("No calibration dataset found for {}. Returning ideal calibration.".format(coord))
            if isinstance(coord, Coordinate.FGBlockOnHICANN):
                return self.ideal_bc.at(c_id)
            else:
                return self.ideal_nc.at(c_id)
        else:
            # return empty calibration if none exists
            self.logger.WARN("No calibration dataset found for {}. Returning empty calibration".format(coord))
            return calibration()

    def get_readout_shift(self, neuron):
        """
        """
        calib = self.get_calibration(neuron)
        if not calib:
            return 0.0

        if calib.exists(pycalibtic.NeuronCalibration.ReadoutShift):
            shift = calib.at(pycalibtic.NeuronCalibration.ReadoutShift).apply(0.0)
            self.logger.TRACE("Readout shift for neuron {0}:{1:.2f}".format(neuron, shift))
        else:
            shift = 0.0
            self.logger.WARN("No readout shift found for neuron {}.".format(neuron))

        return shift

    def apply_calibration(self, value, parameter, coord, use_ideal=False):
        """ Apply calibration to one value. If no calibration is found, use ideal calibration.
            Also, if parameter value is given in a unit that does not support calibration, e.g.
            I_pl given in Ampere instead of microseconds, apply direct translation to DAC.

            Args:
                value: Value in Hardware specific units (e.g. volt, ampere, seconds)
                parameter: neuron_parameter or shared_parameter type
                coord: neuron coordinate
            Returns:
                Calibrated DAC value
                If the calibrated value exceeds the boundaries, it is clipped.
        """
        if not isinstance(value, Unit):
            value = Unit(value)

        calib = self.get_calibration(coord, use_ideal)

        lower_boundary = 0
        if parameter is neuron_parameter.I_pl:
            if not isinstance(value, Second):
                raise TypeError("Calibration for I_pl only valid for seconds.")
            lower_boundary = 4

        if calib.exists(parameter):
            # Check domain
            domain = list(calib.at(parameter).getDomainBoundaries())
            if value.value > max(domain):
                self.logger.WARN("Coord {} value {} larger than domain maximum {}. Clipping value.".format(coord, value.value, max(domain)))
                value.value = max(domain)
            if value.value < min(domain):
                self.logger.WARN("Coord {} value {} smaller than domain minimum {}. Clipping value.".format(coord, value.value, min(domain)))
                value.value = min(domain)
            # Apply calibration
            calib_dac_value = calib.to_dac(value.value, parameter)

            self.logger.TRACE("Calibrated {} parameter {}: {} --> {} DAC".format(coord, parameter.name, value, calib_dac_value))
        else:
            try:
                ideal = self.get_calibration(coord, True)
                calib_dac_value = ideal.at(parameter).apply(value.value)
                self.logger.WARN("Calibration for {} parameter {} not found. Using ideal transformation -> DAC value {}".format(coord, parameter.name, calib_dac_value))
            except RuntimeError, e:
                calib_dac_value = value.toDAC().value
                self.logger.WARN("Calibration for {} parameter {} not found: {}. Using toDAC() value {}".format(coord, parameter.name, e, calib_dac_value))

        if calib_dac_value < lower_boundary:
            self.logger.WARN("Coord {} Parameter {} value {} calibrated to {} is lower than {}. Clipping.".format(coord, parameter, value, calib_dac_value, lower_boundary))
            calib_dac_value = lower_boundary
        if calib_dac_value > 1023:
            self.logger.WARN("Coord {} Parameter {} value {} calibrated to {} is larger than 1023. Clipping.".format(coord, parameter, value, calib_dac_value))
            calib_dac_value = 1023

        return int(round(calib_dac_value))
