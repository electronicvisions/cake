import pycalibtic
import pywrapstdvector
import pylogging
import os

import Coordinate
from pyhalbe.HICANN import neuron_parameter, shared_parameter


def create_pycalibtic_polynomial(coefficients):
    """Create a pycalibtic.Polynomial from a list of coefficients.

    Order: [c0, c1, c2, ...] resulting in c0*x^0 + c1*x^1 + c2*x^2 + ..."""
    # Make standard python list to have the right order
    coefficients = list(coefficients)
    data = pywrapstdvector.Vector_Double(coefficients)
    return pycalibtic.Polynomial(data)


class Calibtic(object):
    logger = pylogging.get("pycake.calibtic")

    def __getstate__(self):
        """ Disable stuff from getting pickled that cannot be pickled.
        """
        odict = self.__dict__.copy()
        del odict['backend']
        del odict['hc']
        del odict['nc']
        del odict['bc']
        del odict['md']
        return odict

    def __setstate__(self, dic):
        # TODO fix for hc, nc, bc, md
        # Initialize logger and calibtic backend when unpickling
        self.path = dic['path']
        dic['backend'] = self.init_backend()
        self.__dict__.update(dic)
        self._load_calibration()

    def __init__(self, path, wafer, hicann):
        self.path = self._make_path(os.path.expanduser(path))
        self.backend = self.init_backend()
        self.wafer = wafer
        self.hicann = hicann
        self._load_calibration()

    def init_backend(self, type='xml'):
        if type == 'xml':
            lib = pycalibtic.loadLibrary('libcalibtic_xml.so')
            backend = pycalibtic.loadBackend(lib)
            backend.config('path', self.path)
            backend.init()
            return backend
        else:
            raise ValueError("unknown backend type")

    def _make_path(self, path):
        if not os.path.isdir(path):
            self.logger.INFO("Creating backend path {}".format(path))
            os.makedirs(path)
        return path

    def clear_calibration(self):
        """ Clears all calibration data.
        """
        name = self.get_calibtic_name()
        fullname = name+".xml"
        fullpath = os.path.join(self.path, fullname)
        if os.path.isfile(fullpath):
            msg = "Clearing calibration data by removing file {}"
            self.logger.INFO(msg.format(fullpath))
            os.remove(fullpath)

    def clear_one_calibration(self, parameter):
        """ Only clears the calibration for one parameter.
        """
        if not self._loaded:
            self._load_calibration()

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
        name = "w{}-h{}".format(int(wafer_id), int(hicann_id))
        return name

    def _load_calibration(self):
        """ Load existing calibration data from backend.
        """
        hc = pycalibtic.HICANNCollection()
        nc = hc.atNeuronCollection()
        bc = hc.atBlockCollection()
        md = pycalibtic.MetaData()

        name = self.get_calibtic_name()

        # Delete all standard entries.
        #TODO: fix calibtic to use proper standard entries
        for nid in range(512):
            nc.erase(nid)
        for bid in range(4):
            bc.erase(bid)

        try:
            self.backend.load(name, md, hc)
            # load existing calibration:
            nc = hc.atNeuronCollection()
            bc = hc.atBlockCollection()
        except RuntimeError, e:
            if e.message != "data set not found":
                raise RuntimeError(e)
            else:
                # backend does not exist
                pass

        self.hc = hc
        self.nc = nc
        self.bc = bc
        self.md = md
        self._loaded = True

    def write_calibration(self, parameter, data):
        """ Writes calibration data
            Coefficients are ordered like this:
            [a, b, c] ==> a*x^0 + b*x^1 + c*x^2 + ...

            Args:
                parameter: which neuron or hicann parameter
                data: dict { coord : coefficients }
        """
        if not self._loaded:
            self._load_calibration()
        name = self.get_calibtic_name()

        for coord, coeffs in data.iteritems():

            if coeffs is None:
                continue

            if isinstance(parameter, shared_parameter) and isinstance(coord, Coordinate.FGBlockOnHICANN):
                collection = self.bc
                cal = pycalibtic.SharedCalibration()
            else:
                collection = self.nc
                cal = pycalibtic.NeuronCalibration()

            # If parameter is V_reset, BUT coordinate is not a block, it is assumed that you want to store readout shifts
            # Readout shifts are stored as parameter
            if parameter is shared_parameter.V_reset and isinstance(coord, Coordinate.NeuronOnHICANN):
                param_id = pycalibtic.NeuronCalibration.VResetShift
                param_name = "readout_shift"
            else:
                param_id = parameter
                param_name = parameter.name

            index = coord.id().value()
            polynomial = create_pycalibtic_polynomial(coeffs)
            if not collection.exists(index):
                collection.insert(index, cal)
            collection.at(index).reset(param_id, polynomial)
            self.logger.TRACE("Resetting coordinate {} parameter {} to {}".format(coord, param_name, polynomial))
        self.backend.store(name, self.md, self.hc)

    def get_calibration(self, coord):
        """
        """
        if not self._loaded:
            self._load_calibration()
        c_id = coord.id().value()

        if isinstance(coord, Coordinate.FGBlockOnHICANN):
            collection = self.bc
        else:
            collection = self.nc

        if collection.exists(c_id):
            return collection.at(c_id)
        else:
            self.logger.WARN("No calibration dataset found for {}.".format(coord))
            return None

    def get_readout_shift(self, neuron):
        """
        """
        if not self._loaded:
            self._load_calibration()

        calib = self.get_calibration(neuron)
        if not calib:
            return 0.0

        if calib.exists(pycalibtic.NeuronCalibration.VResetShift):
            shift = calib.at(pycalibtic.NeuronCalibration.VResetShift).apply(0.0)
            self.logger.TRACE("Readout shift for neuron {0}:{1:.2f}".format(neuron, shift))
        else:
            shift = 0.0
            self.logger.WARN("No readout shift found for neuron {}.".format(neuron))

        return shift

    def apply_calibration(self, dac_value, parameter, coord):
        """ Apply calibration to one value.

            Args:
                dac_value
                parameter
                coord
            Returns:
                Calibrated value if calibration exists. Otherwise,
                the value itself is returned and a warning is given.

                If the calibrated value exceeds the boundaries, it is clipped.
        """
        if not self._loaded:
            self._load_calibration()

        lower_boundary = 0
        if parameter is neuron_parameter.I_pl:
            lower_boundary = 4

        calib = self.get_calibration(coord)
        if not calib:
            return int(round(dac_value))

        if calib.exists(parameter):
            calib_dac_value = calib.at(parameter).apply(dac_value)
            self.logger.TRACE("Calibrated {} parameter {}: {} --> {} DAC".format(coord, parameter.name, dac_value, calib_dac_value))
        else:
            self.logger.WARN("Applying calibration failed: Nothing found for {} parameter {}. Using uncalibrated value {}".format(coord, parameter.name, dac_value))
            calib_dac_value = dac_value

        if calib_dac_value < lower_boundary:
            self.logger.WARN("Coord {} Parameter {} DAC-Value {} Calibrated to {} is lower than {}. Clipping.".format(coord, parameter, dac_value, calib_dac_value, lower_boundary))
            calib_dac_value = lower_boundary
        if calib_dac_value > 1023:
            self.logger.WARN("Coord {} Parameter {} DAC-Value {} Calibrated to {} is larger than 1023. Clipping.".format(coord, parameter, dac_value, calib_dac_value))
            calib_dac_value = 1023

        return int(round(calib_dac_value))
