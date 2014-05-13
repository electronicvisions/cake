"TracesOnDiskDict module, contains TracesOnDiskDict class"""
import tables
import os

import numpy as np

class TracesOnDiskDict(object):
    """A dictonary that stores traces on disc"""
    def __init__(self, directory, filename):
        assert os.path.isdir(directory)
        self.directory = directory
        self.filename = filename
        self.h5file_filter_args = {
                'complib' : 'blosc', 'complevel' : 9}
        self.reverse_keys = {}
        self.__h5file = None

    @property
    def fullpath(self):
        """complete path of the storage file"""
        return os.path.join(self.directory, self.filename)

    @property
    def h5file(self):
        """file object, will be lazy opened"""
        if self.__h5file is None:
            filters = tables.Filters(**self.h5file_filter_args)
            self.__h5file = tables.openFile(
                    self.fullpath, mode="w", title="TracesOnDisk",
                    filters=filters)
        return self.__h5file

    @property
    def root(self):
        """root of the hdf5 file"""
        return self.h5file.root

    def get_key(self, key):
        """Generates a uniqeu key from a coordinate"""
        new_key = "trace_{:0>3}".format(key.id().value())
        self.reverse_keys[new_key] = key
        return new_key

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['hicann']
        return odict

    def __setstate__(self, dic):
        coord_hicann = dic['coord_hicann']
        wafer = dic['wafer']
        dic['hicann'] = wafer[coord_hicann]
        self.__dict__.update(dic)

    def __len__(self):
        return self.root._v_nchildren

    def __delitem__(self, key):
        key = self.get_key(key)
        if getattr(self.root, key, None):
            self.h5file.removeNode(getattr(self.root, key))
        del self.reverse_keys[key]

    def __getitem__(self, key):
        try:
            array = getattr(self.root, self.get_key(key))
            return np.array(array)
        except tables.NoSuchNodeError:
            raise KeyError(key)

    def __setitem__(self, key, arr):
        assert isinstance(arr, np.ndarray)
        key = self.get_key(key)
        if getattr(self.root, key, None):
            self.h5file.removeNode(getattr(self.root, key))
        atom = tables.Atom.from_dtype(arr.dtype)
        tarr = self.h5file.createCArray(self.root, key, atom, arr.shape)
        tarr[:] = arr

    def __iter__(self):
        for value in self.reverse_keys.itervalues():
            yield value

    def itervalues(self):
        """mimics itervalues from dict"""
        for dataset in self.root:
            yield self.reverse_keys[dataset.name], np.array(dataset)

    def iterkeys(self):
        """mimics iterkeys from dict"""
        for value in self.reverse_keys.itervalues():
            yield value

    def iteritems(self):
        """mimics iteritems from dict"""
        for dataset in self.root:
            yield self.reverse_keys[dataset.name], np.array(dataset)

    def keys(self):
        """mimics keys from dict"""
        return [k for k in self]

    def values(self):
        """mimics values from dict"""
        return [v for v in self.itervalues()]
