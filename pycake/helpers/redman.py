import pyredman as redman


def init_backend(type='xml', path='/afs/kip.uni-heidelberg.de/user/mkleider/tmp/redman-xml-backend'):
    # FIXME hardcoded path
    if type == 'xml':
        lib = redman.loadLibrary('libredman_xml.so')
        backend = redman.loadBackend(lib)
        backend.config('path', path)
        backend.init()
        return backend
    else:
        raise ValueError("unknown backend type")
