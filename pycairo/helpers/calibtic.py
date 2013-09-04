import pycalibtic


def create_pycalibtic_polynomial(coefficients):
    """Create a pycalibtic.Polynomial from a list of coefficients.

    Order: [c0, c1, c2, ...] resulting in c0*x^0 + c1*x^1 + c2*x^2 + ..."""
    data = pycalibtic.vector_less__double__greater_()
    for i in coefficients:
        data.append(i)
    return pycalibtic.Polynomial(data)
