"""Weibull 5 method"""
import numpy as np
import scipy.optimize

import uclass.statistics.weibull


class Weibull5:
    """Weibull 5"""
    def __init__(self, hf, class_match="M"):
        """Constructor

        Parameters
        ----------
        hf : array-like
            List of hit factors.
            
        class_match : str
            The target class percentile to match.
            Choose from ["GM", "M", "A"].
            Defaults "M", Top 5th percentile is M class.
            1st percentile and 15th percentile for GM and A. 
        """
        self.hf = hf
        if class_match == "GM":
            self.percentile_match = 0.99
            self.hhf_multiplier = 1 / 0.95
        elif class_match == "M":
            self.percentile_match = 0.95
            self.hhf_multiplier = 1 / 0.85
        elif class_match == "A":
            self.percentile_match = 0.85
            self.hhf_multiplier = 1 / 0.75
        else:
            raise ValueError(f"Class match {class_match} not supported")
        
    @property
    def hf(self):
        """list of hit factor"""
        return self._hf

    @hf.setter
    def hf(self, _hf):
        """hf.setter"""
        self._hf = _hf

    @property
    def percentile_match(self):
        """Percentile to match"""
        return self._percentile_match

    @percentile_match.setter
    def percentile_match(self, _percentile_match):
        """percentile_match.setter"""
        self._percentile_match = _percentile_match

    @property
    def hhf_multiplier(self):
        """HHF multiplier"""
        return self._hhf_multiplier

    @hhf_multiplier.setter
    def hhf_multiplier(self, _hhf_multiplier):
        """hhf_multiplier.setter"""
        self._hhf_multiplier = _hhf_multiplier

    def get_hhf(self):
        """Get hhf"""
        weibull = self.fit_weibull()
        percentile_hf = weibull.quantile(self.percentile_match)
        hhf = percentile_hf * self.hhf_multiplier
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
            nll_ = -np.sum(np.log(likelihood))
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

        return weibull
        
        
