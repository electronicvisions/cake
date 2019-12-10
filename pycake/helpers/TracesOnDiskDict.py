"TracesOnDiskDict module, contains TracesOnDiskDict class"""
import tables
import os
import collections

import numpy as np
import pandas

class TracesOnDiskDict(collections.MutableMapping):
    """A dictonary that stores traces on disc"""
    def __init__(self, directory, filename):
        assert os.path.isdir(directory)
        self.directory = directory
        self.filename = filename
        self.h5file_filter_args = {
                'complib' : 'blosc', 'complevel' : 9}
        self.reverse_keys = {}
        self._h5file = None

    def __del__(self):
        self.close()

    @property
    def fullpath(self):
        """complete path of the storage file"""
        return os.path.join(self.directory, self.filename)

    @property
    def h5file(self):
        """file object, will be lazy opened"""
        if self._h5file is None:
            filters = tables.Filters(**self.h5file_filter_args)
            try:
                self._h5file = tables.open_file(
                        self.fullpath, mode="a", title="TracesOnDisk",
                        filters=filters)
            except IOError:
                # The file might be only read only, in this case try to open
                # as read only
                self._h5file = tables.open_file(
                        self.fullpath, mode="r", title="TracesOnDisk",
                        filters=filters)
        return self._h5file

    def close(self):
        """closes the underlying file object"""
        if self._h5file:
            self._h5file.close()
        self._h5file = None

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
        new_key = "trace_{:0>3}".format(key.toEnum().value())
        self.reverse_keys[new_key] = key
        return new_key

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['_h5file']
        return odict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5file = None

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
        tarr = self.h5file.create_carray(self.root, key, atom, arr.shape)
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
            self.h5file.remove_node(getattr(self.root, key), recursive=True)
        group = self.h5file.create_group(self.root, key)
        for dkey, arr in arrays.iteritems():
            assert isinstance(arr, np.ndarray)
            atom = tables.Atom.from_dtype(arr.dtype)
            tarr = self.h5file.create_carray(group, dkey, atom, arr.shape)
            tarr[:] = arr

    def get_trace(self, key, item='v'):
        """
        Returns a specific trace from the dictionary assoiciated with key.
        """
        try:
            group = getattr(self.root, self.get_key(key))
            return np.array(getattr(group, item))
        except tables.NoSuchNodeError:
            raise KeyError(key)

class PandasRecordsOnDiskDict(RecordsOnDiskDict):
    """A dictonary that stores traces on disc, using pandas HDFStore

    This dictonary expects either a pandas dataframe as value or a dictionary
    with numpy arrays of equal length. The entry 't', will be used as index
    of the DataFrame.
    """

    @property
    def h5file(self):
        """file object, will be lazy opened"""
        if self._h5file is None:
            self._h5file = pandas.HDFStore(
                self.fullpath, mode="a", **self.h5file_filter_args)
        return self._h5file

    root = None

    def get_key(self, key):
        """Generates a uniqeu key from a coordinate"""
        new_key = "/trace_{:0>3}".format(key.toEnum().value())
        self.reverse_keys[new_key] = key
        return new_key

    def __len__(self):
        return len(self.h5file)

    def __delitem__(self, key):
        del self.h5file[self.get_key(key)]

    def __getitem__(self, key):
        return self.h5file[self.get_key(key)]

    def __setitem__(self, key, arrays):
        if isinstance(arrays, dict):
            times = arrays.pop('t')
            arrays = pandas.DataFrame(arrays, index=times)
        elif not isinstance(arrays, pandas.DataFrame):
            raise TypeError("Invalid data type")
        self.h5file[self.get_key(key)] = arrays

    def __iter__(self):
        for value in self.reverse_keys.itervalues():
            yield value

    def get_trace(self, key, item='v'):
        """
        Returns a specific trace from the dictionary assoiciated with key.
        """
        try:
            group = getattr(self.root, self.get_key(key))
            return np.array(getattr(group, item))
        except tables.NoSuchNodeError:
            raise KeyError(key)
