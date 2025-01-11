"""Weibull distribution class"""
import numpy as np
import scipy.special


class Weibull:
    """Weibull distribution class"""
    def __init__(self, lam, k):
        """Constructor
        
        Parameters
        ----------
        lam : float
            Scale parameter.
        k : float
            shape parameter.
        """
        self.lam = lam
        self.k = k

    @property
    def lam(self):
        """Scale parameter"""
        return self._lam

    @lam.setter
    def lam(self, _lam):
        """lam.setter"""
        self._lam = _lam

    @property
    def k(self):
        """Shape parameter"""
        return self._k

    @k.setter
    def k(self, _k):
        """k.setter"""
        self._k = _k

    @property
    def mean(self):
        """Mean"""
        gamma = scipy.special.gamma(1+1/self.k)
        _mean = self.lam * gamma
        return _mean

    @property
    def mode(self):
        """Mode"""
        _mode = self.lam*((self.k-1)/(self.k))**(1/self.k)
        return _mode

    @property
    def median(self):
        """Median"""
        _median = self.lam * np.log(2)**(1/self.k)
        return _median 

    @property
    def kurtosis(self):
        """Kurtosis"""
        gamma4 = scipy.special.gamma(1+4/self.k)
        _kurtosis = (self.lam**4 * gamma4
                     - 4 * self.skewness * self.std**3 * self.mean
                     - 6 * self.mean**2 * self.std**2
                     - self.mean**4)
        _kurtosis /= self.std**4
        _kurtosis -= 3
        return _kurtosis 

    @property
    def skewness(self):
        """Skewness"""
        gamma3 = scipy.special.gamma(1+3/self.k)
        _skewness = gamma3*self.lam**3 - 3*self.mean*self.std**2 - self.mean**3
        _skewness /= self.std**3
        return _skewness

    @property
    def variance(self):
        """Variance"""
        gamma2 = scipy.special.gamma(1+2/self.k)
        gamma1 = scipy.special.gamma(1+1/self.k)
        _variance = self.lam**2 * (gamma2 - gamma1**2)
        return _variance

    @property
    def std(self):
        """Standard deviation"""
        return self.variance**.5

    def pdf(self, x):
        """Probability density function
        
        Parameters
        ----------
        x : array-like
            The random variable

        Returns
        -------
        array-like
            The probability density function.
        """
        k = self.k
        lam = self.lam
        _pdf = (k/lam) * (x/lam)**(k-1) * np.exp(-(x/lam)**k)
        return _pdf

    def cdf(self, x):
        """Cumulative distribution function
        
        Parameters
        ----------
        x : array-like
            The random variable

        Returns
        -------
        array-like
            The cumulative distribution function
        """
        k = self.k
        lam = self.lam
        _cdf = 1-np.exp(-(x/lam)**k)
        return _cdf
