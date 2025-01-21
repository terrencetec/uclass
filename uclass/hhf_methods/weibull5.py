"""Weibull 5 method"""
import numpy as np
import scipy.optimize

import uclass.statistics.weibull


class Weibull5:
    """Weibull 5

    Notes
    -----
    This method fits a Weibull distribution to the hit factor data.
    The high hit factor is then defined as
    `weibull.quantile(percentile) / percentage`,
    with default `percentile = 0.95` and `percentage = 0.85`,
    i.e. Top 5 percent shooters are at least M class.
    """
    def __init__(self, hf, percentile=0.95, percentage=0.85):
        """Constructor

        Parameters
        ----------
        hf : array-like
            List of hit factors.

        percentile : float, optional
            The percentile to match a certain hit factor percentage.
            Defaults 0.95.
        percentage : float, optional
            The hit factor percentage (in fraction) of the percentile.
            Defaults 0.85.
        """
        self.hf = hf
        self.percentile = percentile
        self.percentage = percentage
        self.weibull = None

    @property
    def hf(self):
        """list of hit factor"""
        return self._hf

    @hf.setter
    def hf(self, _hf):
        """hf.setter"""
        self._hf = _hf

    @property
    def percentile(self):
        """Percentile to match"""
        return self._percentile

    @percentile.setter
    def percentile(self, _percentile):
        """percentile.setter"""
        self._percentile = _percentile

    @property
    def percentage(self):
        """Percentage of the percentile"""
        return self._percentage

    @percentage.setter
    def percentage(self, _percentage):
        """percentage.setter"""
        self._percentage = _percentage

    def get_hhf(self, percentile=None, percentage=None):
        """Get high hit factor from match percentile and percentage

        Parameters
        ----------
        Percentile : float
            The percentile to match
        Percentage : float
            The hit factor percentage (in fraction) of the percentile.

        Returns
        -------
        hhf : float
            The high hit factor
        """
        if percentile is not None:
            self.percentile = percentile
        if percentage is not None:
            self.percentage = percentage
        percentile = self.percentile
        percentage = self.percentage
        if self.weibull is None:
            self.fit_weibull()
        percentile_hf = self.weibull.quantile(percentile)
        hhf = percentile_hf / percentage
        return hhf

    def fit_weibull(self, lam0=None, k0=3.6):
        """Fit weibull

        Parameters
        ----------
        lam0 : float, Optional.
            Initial guess of the scale parameter
            Defaults to be the mean of the samples.
        k0 : float, optional
            Initial guess of the shape parameter
            Defaults 3.6
        
        Returns
        -------
        weibull : uclass.statistics.weibull.Weibull
        """
        # TODO Consider putting this function elsewhere.
        def nll(params, x):
            """Negative log likelihood

            Parameters
            ----------
            params : array
                Parameters of the PDF.
            x : array-like
                Observed values of the random variable.
            pdf : func(x, *params) -> float
            """
            lam, k = params
            weibull = uclass.statistics.weibull.Weibull(lam=lam, k=k)
            likelihood = weibull.pdf(x)
            nll_ = -np.mean(np.log(likelihood))
            return nll_

        samples = self.hf
        
        if lam0 is None:
            # lam0 = np.mean(samples)
            lam0 = np.median(samples) / np.log(2)**(1/k0)
        x0 = [lam0, k0]

        res = scipy.optimize.minimize(
            nll, x0=x0, args=samples, method="nelder-mead")
        
        lam, k = res.x
        weibull = uclass.statistics.weibull.Weibull(lam, k)

        self.weibull = weibull

        return weibull
        
        
