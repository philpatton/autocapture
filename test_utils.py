import numpy as np

from src.utils import freeman_tukey

def test_freeman_tukey():
    o = [5,5]
    e = [4,4]
    D = freeman_tukey(o, e)
    should_be = 0.11
    assert np.isclose(D, should_be, atol=0.01)