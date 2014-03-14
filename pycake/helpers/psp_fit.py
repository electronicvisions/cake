import numpy as np
from scipy import optimize
import pylogging

logger = pylogging.get("pycake.helper.psp_fit")


def fit(psp_shape, time, voltage, error_estimate, maxcall=1000,
        maximal_red_chi2=50000.0, fail_on_negative_cov=None, jacobian=None):
    """
    np.p_shape : object
        PSPShape instance

    time : numpy.ndarray of floats
        numpy array of data acquisition times

    voltage : numpy.ndarray
        numpy array of voltage values

    error_estimate : float
        estimate for the standard deviation of an individual data np.int.

    maxcall : int
        maximal number of calls to the fit routine

    fail_on_negative_cov : list of int

    returns : tuple
        (success,
         fit_results
         error_estimates
         chi2_per_dof)
    """
    assert time.shape == voltage.shape

    parnames = psp_shape.parameter_names()
    initial_values = psp_shape.initial_fit_values(time, voltage)

    scale = np.ones(len(parnames), dtype=np.float64)
    scale_value = 1.0/np.mean( (initial_values['tau_1'], initial_values['tau_2']) )
    for param in ('start', 'tau_1', 'tau_2'):
        scale[parnames.index(param)] = scale_value

    f = lambda param: (psp_shape(time, *param) - voltage)
    x0 = [initial_values[key] for key in parnames]
    if jacobian:
        j = lambda param: jacobian(time, *param)
        resultparams, cov_x, info, fit_msg, ier = optimize.leastsq(
            f, x0, Dfun=j, col_deriv=True,
            full_output=1, maxfev=maxcall,
            diag=scale
            )
    else:
        # If epsfcn is not set estimating of the jacobian will fail
        resultparams, cov_x, info, fit_msg, ier = optimize.leastsq(
            f, x0,
            full_output=1, maxfev=maxcall,
            epsfcn=1e-5,
            diag=scale
        )

    ndof = len(time) - len(parnames)
    fit_voltage = psp_shape(time, *resultparams)
    red_chi2 = sum((fit_voltage - voltage) ** 2) / (error_estimate ** 2 * ndof)

    logger.TRACE("Raw fit result: {}: {}".format(ier, fit_msg))
    logger.TRACE("                Covariance matrix: {}".format(cov_x))
    logger.TRACE("                Reduced Chi**2: {}".format(red_chi2))

    # return code 1 to 4 indicates success
    success = True
    if not ier in [1, 2, 3, 4]:
        success = False
        logger.INFO("Fit rejected by fitting error: {} (code: {})".format(
            fit_msg, ier))

    if cov_x is None:
        success = False
        logger.INFO("Fit rejected by missing covariance matrix")
    else:
        if fail_on_negative_cov is not None:
            negative = np.logical_and(np.diag(cov_x) < 0, fail_on_negative_cov)
            names = [ n for n, ok in zip(parnames, negative) if ok ]
            if np.any(negative):
                logger.INFO("Fit rejected by negative covariance of"
                    "parameters " + ",".join(names))
                success = False
        cov_x *= error_estimate ** 2

    if red_chi2 > maximal_red_chi2:
        logger.INFO("Fit rejected by chi2: {} (limit: {})".format(
            red_chi2, maximal_red_chi2))
        success = False

    processed, proocessed_cov = psp_shape.process_fit_results(
        resultparams,
        cov_x)
    return success, processed, proocessed_cov, red_chi2
