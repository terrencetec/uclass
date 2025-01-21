"""Test uclass.hhf_methods.ppregress"""
import pickle

import numpy as np

import uclass


def test_ppregress():
    """Test PPRegress()"""
    with open("tests/data/co_hf_sample.pkl", "rb") as f:
        hf_sample = pickle.load(f)
    with open("tests/data/co_hhf_sample.pkl", "rb") as f:
        hhf_sample = pickle.load(f)
    hf = np.loadtxt("tests/data/co_23-01.txt")

    ppregress = uclass.PPRegress(hf, hf_sample, hhf_sample)
    hhf = ppregress.get_hhf()
    hhf_true = 10.85615024
    
    assert np.isclose(hhf, hhf_true)
