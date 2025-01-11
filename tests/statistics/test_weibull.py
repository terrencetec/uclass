"""Test uclass.statistics.weibull"""
import unittest

import numpy as np

import uclass.statistics.weibull


class TestWeibull(unittest.TestCase):
    """Test weibull"""
    def setUp(self):
        """Constructor"""
        k = 3.6
        lam = 5
        self.weibull = uclass.statistics.weibull.Weibull(lam=lam, k=k)

    def test_mean(self):
        """Test mean"""
        mean = self.weibull.mean
        mean_true = 4.5055
        self.assertTrue(np.isclose(mean, mean_true))

    def test_mode(self):
        """Test mode"""
        mode = self.weibull.mode
        mode_true = 4.56785076
        self.assertTrue(np.isclose(mode, mode_true))

    def test_median(self):
        """Test median"""
        median = self.weibull.median
        median_true = 4.5160095
        self.assertTrue(np.isclose(median, median_true))

    def test_variance(self):
        """Test variance"""
        variance = self.weibull.variance
        variance_true = 1.932382
        self.assertTrue(np.isclose(variance, variance_true))

    def test_std(self):
        """Test std"""
        std = self.weibull.std
        std_true = 1.3901014
        self.assertTrue(np.isclose(std, std_true))

    def test_skewness(self):
        """Test skewness"""
        skewness = self.weibull.skewness
        skewness_true = 0.0005629389
        self.assertTrue(np.isclose(skewness, skewness_true))

    def test_kurtosis(self):
        """Test kurtosis"""
        kurtosis = self.weibull.kurtosis
        kurtosis_true = -0.2832548
        self.assertTrue(np.isclose(kurtosis, kurtosis_true))
