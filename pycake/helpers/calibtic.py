import pycalibtic
import pywrapstdvector
import pylogging
import os
import time

import pyhalbe.Coordinate as Coordinate
from pyhalbe.HICANN import neuron_parameter, shared_parameter

class Calibtic(object):
    def __getstate__(self):
        """ Disable stuff from getting pickled that cannot be pickled.
        """
        odict = self.__dict__.copy()
        del odict['logger']
        del odict['backend']
        return odict

    def __setstate__(self, dic):
        # Initialize logger and calibtic backend when unpickling
        self.path = dic['path']
        dic['logger'] = pylogging.get("pycake.calibrationrunner")
        dic['backend'] = self.init_backend()
        self.__dict__.update(dic)

    def __init__(self, path, wafer, hicann):
        self.logger = pylogging.get("pycake.calibtic")
        self.path = self.make_path(path)
        self.backend = self.init_backend()
        self.wafer = wafer
        self.hicann = hicann

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
            self.logger.INFO("{} - Clearing calibration data by removing file {}".format(time.asctime(), fullpath))
            os.remove(fullpath)

    def create_pycalibtic_polynomial(self, coefficients):
        """Create a pycalibtic.Polynomial from a list of coefficients.

        Order: [c0, c1, c2, ...] resulting in c0*x^0 + c1*x^1 + c2*x^2 + ..."""
        data = pywrapstdvector.Vector_Double(coefficients)
        return pycalibtic.Polynomial(data)

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

        # Delete all standard entries. TODO: fix calibtic to use proper standard entries
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

        return (hc, nc, bc, md)

    def write_calibration(self, parameter, coord, coeffs):
        """ Writes calibration data
            Coefficients are ordered like this:
            [a, b, c] ==> a*x^0 + b*x^1 + c*x^2 + ...
        """
        name = self.get_calibtic_name()
        hc, nc, bc, md = self.load_calibration()

        if isinstance(parameter, shared_parameter) and isinstance(coord, Coordinate.FGBlockOnHICANN):
            collection = bc
            cal = pycalibtic.SharedCalibration()
        else:
            collection = nc
            cal = pycalibtic.NeuronCalibration()
        
        self.logger.TRACE("Writing {} calibration to backend.".format(parameter.name))
        index = coord.id().value()
        self.logger.TRACE("Creating polynomial with coefficients {}".format(coeffs))
        polynomial = self.create_pycalibtic_polynomial(coeffs)
        if not collection.exists(index):
            collection.insert(index, cal)
        collection.at(index).reset(parameter, polynomial)
        self.logger.TRACE("Resetting parameter {} to polynomial {}".format(parameter.name, polynomial))

        self.backend.store(name, md, hc)

    def get_calibration(self, coord):
        """ Returns calibration for a specific coordinate
            Equivalent to calling e.g. nc.at(coordinate_id)
        """
        hc, nc, bc, md = self.load_calibration()
        c_id = coord.id().value()
        if isinstance(coord, Coordinate.FGBlockOnHICANN):
            return bc.at(c_id)
        else:
            return nc.at(c_id)
