"""Distribution base class"""


class Distribution:
    """Distribution base class"""
    def __init__(self):
        """Constructor"""
        pass

    @property
    def mean(self):
        """Mean"""
        return None

    @property
    def mode(self):
        """Mode"""
        return None

    @property
    def median(self):
        """Median"""
        return None

    @property
    def kurtosis(self):
        """Kurtosis"""
        return None

    @property
    def skewness(self):
        """Skewness"""
        return None

    @property
    def variance(self):
        """Variance"""
        return None

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
        return None

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
        return None



