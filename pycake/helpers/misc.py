import os
import errno


# http://stackoverflow.com/a/600612/1350789
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:  # pragma: no cover
            # unexpected, re-raise exception
            raise
