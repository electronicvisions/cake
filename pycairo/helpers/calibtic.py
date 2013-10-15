import pycalibtic


def init_backend(type='xml', path='/afs/kip.uni-heidelberg.de/user/mkleider/tmp/calibtic-xml-backend'):
    # FIXME hardcoded path
    if type == 'xml':
        lib = pycalibtic.loadLibrary('libcalibtic_xml.so')
        backend = pycalibtic.loadBackend(lib)
        backend.config('path', path)
        backend.init()
        return backend
    else:
        raise ValueError("unknown backend type")


def create_pycalibtic_polynomial(coefficients):
    """Create a pycalibtic.Polynomial from a list of coefficients.

    Order: [c0, c1, c2, ...] resulting in c0*x^0 + c1*x^1 + c2*x^2 + ..."""
    data = pycalibtic.vector_less__double__greater_()
    for i in coefficients:
        data.append(i)
    return pycalibtic.Polynomial(data)
