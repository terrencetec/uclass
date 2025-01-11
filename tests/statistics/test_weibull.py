"""Test uclass.statistics.weibull"""
import numpy as np

import uclass.statistics.weibull


k = 3.6
lam = 5
weibull = uclass.statistics.weibull.Weibull(lam=lam, k=k)


def test_mean():
    """Test mean"""
    mean = weibull.mean
    mean_true = 4.5055
    assert np.isclose(mean, mean_true)


def test_mode():
    """Test mode"""
    mode = weibull.mode
    mode_true = 4.56785076
    assert np.isclose(mode, mode_true)


def test_median():
    """Test median"""
    median = weibull.median
    median_true = 4.5160095
    assert np.isclose(median, median_true)


def test_variance():
    """Test variance"""
    variance = weibull.variance
    variance_true = 1.932382
    assert np.isclose(variance, variance_true)


def test_std():
    """Test std"""
    std = weibull.std
    std_true = 1.3901014
    assert np.isclose(std, std_true)


def test_skewness():
    """Test skewness"""
    skewness = weibull.skewness
    skewness_true = 0.0005629389
    assert np.isclose(skewness, skewness_true)


def test_kurtosis():
    """Test kurtosis"""
    kurtosis = weibull.kurtosis
    kurtosis_true = -0.2832548
    assert np.isclose(kurtosis, kurtosis_true)
     

def test_pdf():
    """Test pdf"""
    x = np.linspace(0, 10, 1024)
    weibull.pdf(x)


def test_cdf():
    """Test cdf"""
    x = np.linspace(0, 10, 1024)
    weibull.cdf(x)
