import os
import time
import shutil
import pyredman
import pyhalco_hicann_v2 as Coordinate
import pylogging

from pyhalco_common import iter_all

class Redman(object):
    logger = pylogging.get("pycake.redman")

    def __getstate__(self):
        """ Disable stuff from getting pickled that cannot be pickled.
        """
        odict = self.__dict__.copy()
        del odict['backend']
        del odict['hicann_with_backend']
        return odict

    def __setstate__(self, dic):
        # Initialize logger and redman backend when unpickling
        self.path = dic['path']
        dic['backend'] = self.init_backend()
        self.__dict__.update(dic)
        self._load_defects()

    def __init__(self, path, hicann):
        self.path = self._make_path(os.path.expanduser(path))
        self.backend = self.init_backend()
        self.hicann = hicann
        self._load_defects()

    def init_backend(self, type='xml'):
        if type == 'xml':
            lib = pyredman.loadLibrary('libredman_xml.so')
            backend = pyredman.loadBackend(lib)
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

    def clear_defects(self):
        """ Clears all defects data.
        """

        self.logger.INFO("clearing defects")

        self.hicann_with_backend.neurons().enable_all()

        fullpath = os.path.join(
            self.path, self.hicann_with_backend.id_for_backend() + ".xml")
        if os.path.isfile(fullpath):
            shutil.copyfile(fullpath, fullpath + time.strftime('.%y%m%d_%H%M%S.bak'))
        self.hicann_with_backend.save()

    def _load_defects(self):
        """ Load existing defects data from backend.
        """

        self.hicann_with_backend = pyredman.HicannWithBackend(self.backend, self.hicann)

        self._loaded = True

    def write_defects(self, coords):
        """ Writes defects
        """
        if not self._loaded:
            self._load_defects()

        for coord in coords:
            if isinstance(coord, Coordinate.NeuronOnHICANN):
                if self.hicann_with_backend.neurons().has(coord):
                    self.logger.INFO("disabling {}".format(coord))
                    self.hicann_with_backend.neurons().disable(coord)
                else:
                    self.logger.INFO("already disabled {}".format(coord))

        self.hicann_with_backend.save()
