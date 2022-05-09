"""Functions for performing statistical tests between two Series of Dataframes of Features.

Testing functions return one p-value indicating the statistical significance of the test result.
"""

import scipy
import spm1d
import numpy as np

def test_kolmogorov_smirnov(series_1, series_2):
    return scipy.stats.kstest(series_1, series_2)

def test_hotellings_t_squared(series_1, series_2):
    if np.array_equal(series_1, series_2): return 1 # otherwise test crashes if there is a perfect correlation
    return spm1d.stats.hotellings2(series_1, series_2)
