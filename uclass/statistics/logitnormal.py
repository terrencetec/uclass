"""Logit-normal distribution"""
import numpy as np
import scipy.special


class LogitNormal:
    """Logit-normal distribution class"""
    def __init__(self, sigma, mu):
        """Constructor
        
        Parameters
        ----------
        sigma : float
            Scale.
        mu : float
            Location.
        """
        self.sigma = sigma
        self.mu = mu

    @property
    def sigma(self):
        """Scale"""
        return self._sigma

    @sigma.setter
    def sigma(self, _sigma):
        """sigma.setter"""
        self._sigma = _sigma

    @property
    def mu(self):
        """Location"""
        return self._mu

    @mu.setter
    def mu(self, _mu):
        """mu.setter"""
        self._mu = _mu

    def pdf(self, x):
        """Probability density function
        
        Parameters
        ----------
        x : array-like
            Random variable.

        Returns
        -------
        array-like
            The probability density function.
        """
        logit = lambda p: np.log(p/(1-p))
        _pdf = (1 / (self.sigma*np.sqrt(2*np.pi))
                * 1 / (x*(1-x))
                * np.exp(-(logit(x)-self.mu)**2/(2*self.sigma)**2))
        return _pdf

    def cdf(self, x):
        """Cumulative distribution function

        Parameters
        ----------
        x : array-like
            Random variable.

        Returns
        -------
        array-like
            The cumulative distribution function.
        """
        logit = lambda p: np.log(p/(1-p))
        _cdf = (1 / 2
                * (1 + scipy.special.erf(
                    (logit(x)-self.mu)/(np.sqrt(2*self.sigma**2)))))
        return _cdf
        

