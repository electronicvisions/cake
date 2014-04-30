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
        self.load_calibration()

    def __init__(self, path, wafer, hicann):
        self.path = self.make_path(os.path.expanduser(path))
        self.backend = self.init_backend()
        self.wafer = wafer
        self.hicann = hicann
        self.load_calibration()

    def init_backend(self, type='xml'):
        if type == 'xml':
            lib = pycalibtic.loadLibrary('libcalibtic_xml.so')
            backend = pycalibtic.loadBackend(lib)
            backend.config('path', self.path)
            backend.init()
            return backend
        else:
            raise ValueError("unknown backend type")

    def make_path(self, path):
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

    def get_calibtic_name(self):
        wafer_id = self.wafer.value()
        hicann_id = self.hicann.id().value()
        name = "w{}-h{}".format(int(wafer_id), int(hicann_id))
        return name

    def load_calibration(self):
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

    def write_calibration(self, parameter, data):
        """ Writes calibration data
            Coefficients are ordered like this:
            [a, b, c] ==> a*x^0 + b*x^1 + c*x^2 + ...

            Args:
                parameter: which neuron or hicann parameter
                data: dict { coord : coefficients }
        """
        # TODO check if loaded, if not: load
        name = self.get_calibtic_name()

        for coord, coeffs in data.iteritems():
            if isinstance(parameter, shared_parameter) and isinstance(coord, Coordinate.FGBlockOnHICANN):
                collection = self.bc
                cal = pycalibtic.SharedCalibration()
            else:
                collection = self.nc
                cal = pycalibtic.NeuronCalibration()

            if parameter is shared_parameter.V_reset and isinstance(coord, Coordinate.NeuronOnHICANN):
                param_id = 21
            else:
                param_id = parameter

            index = coord.id().value()
            polynomial = create_pycalibtic_polynomial(coeffs)
            if not collection.exists(index):
                collection.insert(index, cal)
            collection.at(index).reset(param_id, polynomial)
            self.logger.TRACE("Resetting coordinate {} parameter {} to {}".format(coord, parameter.name, polynomial))
        self.backend.store(name, self.md, self.hc)

    def get_calibrations(self, coords):
        """ Returns one calibration for a specific coordinate
            Equivalent to calling e.g. nc.at(coordinate_id)
        """
        #hc, nc, bc, md = self.load_calibration()

        calibs = {}

        if not isinstance(coords, list):
            coords = [coords]

        for coord in coords:
            c_id = coord.id().value()
            if isinstance(coord, Coordinate.FGBlockOnHICANN):
                calibs[coord] = self.bc.at(c_id)
            else:
                calibs[coord] = self.nc.at(c_id)
        return calibs
