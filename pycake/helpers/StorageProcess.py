"""bla bla"""

import multiprocessing
import Queue
import time
import os
import tempfile
import bz2
import gzip
import shutil
import cPickle
import pylogging


class StorageProcess(object):
    logger = pylogging.get("cake.StorageProcess")
    def __init__(self, compresslevel):
        """
        Args:
            compressionlevel: if < 1 no compression will be used, otherwise
                              it will be given to the bz2 file
        """
        self.process = (None, None)
        self.compresslevel = compresslevel

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['process']
        return odict

    def __setstate__(self, cls_dict):
        cls_dict['process'] = (None, None)
        self.__dict__.update(cls_dict)


    def save_object(self, fullpath, obj):
        """Save the object. It will start an new process and save the
        object state in background. If there is still a process working, it will
        block until its finished"""
        self.join()
        result = multiprocessing.JoinableQueue()
        assert os.path.isdir(os.path.dirname(fullpath))
        process = multiprocessing.Process(target=self.pickle,
                args=(result, fullpath, obj, self.compresslevel))
        process.start()
        self.process = (process, result)

    def join(self):
        """Wait for the storage process to finish writing"""
        process, result_queue = self.process
        self.process = (None, None)
        if process is None:
            return

        process.join()
        try:
            result = result_queue.get(False)
        except Queue.Empty:
            # Note: this might happen, when the OS out-of-memory killer, kills
            # our poor innocent storage process
            self.logger.error("Storage process seems to be lost")
            raise RuntimeError("Storage process died unexpectedly")

        if isinstance(result, Exception):
            raise result
        else:
            return result

    @classmethod
    def pickle(cls, result, filename, obj, compresslevel):
        try:
            tstart = time.time()

            if filename.endswith('.bz2'):
                fopen = bz2.BZ2File
            elif filename.endswith('.gz'):
                fopen = gzip.GzipFile
            elif compresslevel > 0:
                fopen = gzip.GzipFile
                filename += '.gz'
            else:
                fopen = open

            # Note: this is not 100% safe against race condition when calling it
            # from multiple processes!
            tmpfile = tempfile.mktemp(dir=(os.path.dirname(filename)))
            with fopen(tmpfile, 'wb') as outfile:
                cPickle.dump(obj, outfile, cPickle.HIGHEST_PROTOCOL)
            shutil.move(tmpfile, filename)
            cls.logger.INFO("Pickled object in {}s to '{}'".format(
                    int(time.time() - tstart), filename))
            result.put(None)
        except Exception as error:
            cls.logger.error("Error in StorageProcess.pickle: " + str(error))
            result.put(error)
        # Wait for put to finish writing, else there is a Broken Pipe error
        # Check if result is filled, else wait for 1 us
        # cf. https://bugs.python.org/issue35844
        while (result.empty()):
            time.sleep(1e-6)
        result.close()

