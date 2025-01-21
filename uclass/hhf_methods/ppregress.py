"""Percentile-percentage regression method"""
import numpy as np
import scipy.optimize

import uclass.hhf_methods.weibull5


class PPRegress:
    """Percentile-percentage regression method

    Notes
    -----
    This method finds the high hit factor of a classifier stage
    to other stages with known high hit factors by doing a
    regression of the percentile and percentage of the
    Weibull5 method.
    """
    def __init__(self, hf, hf_sample, hhf_sample):
        """Constructor

        Parameters
        ----------
        hf : array-like
            List of hit factors.
        hf_sample : list of arrar-like
            Past hit factors of stages with known high hit factors.
            Each row is historical hit factors of a classifier stage.
        hhf_sample : array-like
            The known high hit factors.
        """
        self.hf = hf
        self.hf_sample = hf_sample
        self.hhf_sample = hhf_sample
        self.percentile = None
        self.percentage = None
        
    @property
    def hf(self):
        """List of hit factors"""
        return self._hf

    @hf.setter
    def hf(self, _hf):
        """hf.setter"""
        self._hf = _hf

    @property
    def hf_sample(self):
        """Past hit factors"""
        return self._hf_sample
    
    @hf_sample.setter
    def hf_sample(self, _hf_sample):
        """hf_sample.setter"""
        self._hf_sample = _hf_sample

    @property
    def hhf_sample(self):
        """Past high hit factors"""
        return self._hhf_sample
    
    @hhf_sample.setter
    def hhf_sample(self, _hhf_sample):
        """hhf_sample.setter"""
        self._hhf_sample = _hhf_sample

    def regress(self):
        """Find best fit percentile and percentage

        Returns
        -------
        percentile : float
            Percentile.
        percentage : float
            Percentage.
        """
        # Fit list of weibulls to historical hit factors
        list_weibull = []
        hf_sample = self.hf_sample
        hhf_sample = self.hhf_sample
        for i in range(len(hhf_sample)):
            weibull5 = uclass.hhf_methods.weibull5.Weibull5(hf_sample[i])
            weibull = weibull5.fit_weibull()
            list_weibull.append(weibull)

        self._list_weibull = list_weibull  # For debug.
        
        # cost
        def cost(params, list_weibull, hhf_sample):
            """Cost function"""
            percentile, percentage = params
            
            hhf_estimate = []

            for weibull in list_weibull:
                hhf = weibull.quantile(percentile) / percentage
                hhf_estimate.append(hhf)

            hhf_estimate = np.array(hhf_estimate)
            hhf_sample = np.array(hhf_sample)
            error = np.mean(np.abs(np.log(hhf_estimate/hhf_sample))**2)

            return error
        
        # Regress
        bounds = [(1e-6, 1-1e-6), (1e-6, 1-1e-6)]
        res = scipy.optimize.differential_evolution(
            cost, bounds=bounds, args=(list_weibull, hhf_sample), rng=123)

        percentile, percentage = res.x

        self.percentile = percentile
        self.percentage = percentage

        return percentile, percentage

    def get_hhf(self):
        """Get high hit factor
        
        Returns
        -------
        hhf : float
            The high hit factor.
        """
        # Best fit percentile and percentage
        if self.percentage is None or self.percentile is None:
            self.regress()

        # Fit weibull for hfs.
        weibull5 = uclass.hhf_methods.weibull5.Weibull5(self.hf)
        weibull = weibull5.fit_weibull()
        
        self._weibull = weibull  # For debug.

        hhf = weibull5.get_hhf(self.percentile, self.percentage)

        return hhf


