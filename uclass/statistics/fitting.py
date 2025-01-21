"""Probability distribution fitting class"""
import numpy as np


def negative_log_likelihood(params, x, pdf):
    """Cost function for maximum likelihood estimation.
    
    Parameters
    ----------
    params : array-like
        Parameters of the probability distribution function
    x : array-like
        Samples of random variables.
    pdf : func
        Probability distribution function.

    Returns
    -------
    float
        Mean negative log likelihood.
    """
    nll = -np.log(pdf(x, *params))
    return np.mean(nll)


def fit_distribution(pdf, x, params0):
    """Fit distribution to samples of random variables

    Parameters
    ----------
    pdf : func
        Probability distribution function.
    x : array-like
        Samples of random variables.
    params0 : array-like
        Iniital guess of the PDF parameters.

    Returns
    -------
    res : scipy.optimize.OptimizeResult
    """
    res = scipy.optimize.minimize(
        negative_log_likehood, x0=params0, args=(x, pdf),
        method="Nelder-Mead")
    return res

