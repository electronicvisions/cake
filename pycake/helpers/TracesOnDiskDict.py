"TracesOnDiskDict module, contains TracesOnDiskDict class"""
import tables
import os
import collections

import numpy as np

class TracesOnDiskDict(collections.MutableMapping):
    """A dictonary that stores traces on disc"""
    def __init__(self, directory, filename):
        assert os.path.isdir(directory)
        self.directory = directory
        self.filename = filename
        self.h5file_filter_args = {
                'complib' : 'blosc', 'complevel' : 9}
        self.reverse_keys = {}
        self.__h5file = None

    def __del__(self):
        self.close()

    @property
    def fullpath(self):
        """complete path of the storage file"""
        return os.path.join(self.directory, self.filename)

    @property
    def h5file(self):
        """file object, will be lazy opened"""
        if self.__h5file is None:
            filters = tables.Filters(**self.h5file_filter_args)
            try:
                self.__h5file = tables.openFile(
                        self.fullpath, mode="a", title="TracesOnDisk",
                        filters=filters)
            except IOError:
                # The file might be only read only, in this case try to open
                # as read only
                self.__h5file = tables.openFile(
                        self.fullpath, mode="r", title="TracesOnDisk",
                        filters=filters)
        return self.__h5file

    def close(self):
        """closes the underlying file object"""
        if self.__h5file:
            self.__h5file.close()
        self.__h5file = None

    @property
    def root(self):
        """root of the hdf5 file"""
        return self.h5file.root

    def update_directory(self, directory):
        """Updates storage path, useful after unpickling"""
        assert os.path.isdir(directory)
        self.close()
        self.directory = directory

    def get_key(self, key):
        """Generates a uniqeu key from a coordinate"""
        new_key = "trace_{:0>3}".format(key.id().value())
        self.reverse_keys[new_key] = key
        return new_key

    def __getstate__(self):
        odict = self.__dict__.copy()
        odict['_TracesOnDiskDict__h5file'] = None
        return odict

    def __len__(self):
        return self.root._v_nchildren

    def __delitem__(self, key):
        _key = self.get_key(key)
        if getattr(self.root, _key, None):
            self.h5file.removeNode(getattr(self.root, _key))
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

class RecordsOnDiskDict(TracesOnDiskDict):
    """A dictonary that stores traces on disc.

    The expected key is a neuron Coordinte, the expected value is a dictonary
    holding the traces as numpy arrays, e.g.: {
        't': numpy.array([0,1]),
        'v': numpy.array([0.9,1.0]),
        'spikes': numpy.array()
    }
    """

    def __getitem__(self, key):
        try:
            group = getattr(self.root, self.get_key(key))
            return dict((v.name, np.array(v)) for v in group)
        except tables.NoSuchNodeError:
            raise KeyError(key)

    def __setitem__(self, key, arrays):
        key = self.get_key(key)
        if getattr(self.root, key, None):
            self.h5file.removeNode(getattr(self.root, key), recursive=True)
        group = self.h5file.createGroup(self.root, key)
        for dkey, arr in arrays.iteritems():
            assert isinstance(arr, np.ndarray)
            atom = tables.Atom.from_dtype(arr.dtype)
            tarr = self.h5file.createCArray(group, dkey, atom, arr.shape)
            tarr[:] = arr

