"""Test Weibull5 method"""
import numpy as np

import uclass


def test_weibull5():
    """Test weibull5"""
    hf = np.loadtxt("tests/data/co_23-01.txt")
    weibull5 = uclass.Weibull5(hf)
    hhf = weibull5.get_hhf(.95, .85)
    hhf_true = 10.5804297
    assert np.isclose(hhf, hhf_true)
