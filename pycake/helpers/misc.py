import os
import errno
import collections


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


# https://stackoverflow.com/a/3233356
def nested_update(d, u):
    """Update nested dictionary with other nested dictionary.

    Args:
        d: original dictionary
        u: update

    Returns:
        Updated original dictionary.
        Note that the original dictionary is modified, too.
    """
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = nested_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d
