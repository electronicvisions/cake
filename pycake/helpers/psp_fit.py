import numpy as np
from scipy import optimize
import pylogging

logger = pylogging.get("pycake.helper.psp_fit")


def fit(psp_shape, time, voltage, error_estimate, maxcall=1000,
        maximal_red_chi2=50000.0, fail_on_negative_cov=None):
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
        (fit_results
         error_estimates
         chi2_per_dof
         success)
    """
    assert time.shape == voltage.shape

    initial_values = psp_shape.initial_fit_values(time, voltage)

    resultparams, cov_x, _, fit_msg, ier = optimize.leastsq(
        lambda param: (psp_shape(time, *param) - voltage),
        [initial_values[key] for key in psp_shape.parameter_names()],
        full_output=1, maxfev=maxcall)

    ndof = len(time) - len(psp_shape.parameter_names())
    fit_voltage = psp_shape(time, *resultparams)
    red_chi2 = sum((fit_voltage - voltage) ** 2) / (error_estimate ** 2 * ndof)

    success = True
    if not cov_x is None:
        fail_neg = np.any(np.diag(cov_x) < 0)
        if fail_on_negative_cov is not None:
            success = np.any(np.logical_and(np.diag(cov_x) < 0,
                            fail_on_negative_cov))

        cov_x *= error_estimate ** 2
    # else:
    #     success = True

    logger.TRACE("Raw fit result: {}: {}".format(ier, fit_msg))
    logger.TRACE("                Covariance matrix: {}".format(cov_x))
    logger.TRACE("                Reduced Chi**2: {}".format(red_chi2))

    # return code 1 to 4 indicates success
    if not ier in [1, 2, 3, 4]:
        logger.INFO("Fit rejected by fitting error: {} (code: {})".format(
            fit_msg, ier))
        success = False
    if red_chi2 > maximal_red_chi2:
        logger.INFO("Fit rejected by chi2: {} (limit: {})".format(
            red_chi2, maximal_red_chi2))
        success = False

    processed, proocessed_cov = psp_shape.process_fit_results(
        resultparams,
        cov_x)
    return success, processed, proocessed_cov, red_chi2
