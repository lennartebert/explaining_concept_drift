"""Functions for performing statistical tests between two Series of Dataframes of Features.

Testing functions return one p-value indicating the statistical significance of the test result.

TODO comment functions
"""

import scipy
import spm1d
import numpy as np
import pandas as pd

def _all_equal(series_1, series_2):
    """Return whether all values in series are equal or different"""
    
    # do both series have the same classes?
    series_1_value_counts = series_1.value_counts()
    series_2_value_counts = series_2.value_counts()
    
    if set(series_1_value_counts.index) != set(series_2_value_counts.index):
        return False

    # check if both series have the exact same count of unique values
    if series_1_value_counts.equals(series_2_value_counts):
        return True
    
    return False

def _all_dissimilar(series_1, series_2):
    """Return whether all series 2 has no values that exist in series 1"""
    set_series_1 = set(series_1.values)
    set_series_2 = set(series_2.values)
    
    if len(set_series_1) == len(set_series_1 - set_series_2): return True
    
    return False

def test_kolmogorov_smirnov(series_1, series_2):
    test_statistics, p_value = scipy.stats.kstest(series_1, series_2)
    return p_value

def test_hotellings_t_squared(series_1, series_2):
    if _all_equal(series_1, series_2): return 1 # otherwise test crashes if there is a perfect correlation
    return spm1d.stats.hotellings2(series_1, series_2)

def test_chi_squared(series_1, series_2):
    if _all_equal(series_1, series_2): return 1
    if _all_dissimilar(series_1, series_2): return 0
    
    # series_1 and series 2 need to have the same number of samples for chi_square to work
    # We test for this criteria, handle it and return a message to the user if this is not the case.
    length_series_1 = len(series_1)
    length_series_2 = len(series_2)
    if length_series_1 != length_series_2:
        print(f"series_1 and series_2 do not have the same number of samples: {length_series_1} and {length_series_2}! We'll handle it by undersampling the series with more observations.")
        
    # handle the case that series_1 is longer than series_2:
    if length_series_1 > length_series_2:
        series_1 = series_1.sample(n=length_series_2, replace=False)
        length_series_1 = len(series_1)
    # handle the case that series_2 is longer than series_1:
    elif length_series_2 > length_series_1:
        series_2 = series_2.sample(n=length_series_1, replace=False)
        length_series_2 = len(series_2)
    
    # get value counts for each series
    value_counts_1 = series_1.value_counts()
    value_counts_2 = series_2.value_counts()
    
    # make sure that both have the same legth
    # value_counts_df = value_counts_1.to_frame().join(value_counts_2, how='left', rsuffix='_') # left join prevents expected values to be 0
    
    # value_counts_df = value_counts_1.to_frame()
    value_counts_df = pd.concat([value_counts_1, value_counts_2], axis=1)
    
    # rename the columns
    value_counts_df.columns = ['expected', 'observed']
    
    # replace nan with 0
    value_counts_df = value_counts_df.fillna(0)
    
    # compute the observed frequencies in each category
    distribution_expected = value_counts_df['expected']
    distribution_observed = value_counts_df['observed']
    
#     # get values as probabilities (need to sum up to 1)
#     distribution_expected_p = distribution_expected / distribution_expected.sum()
#     distribution_observed_p = distribution_observed / distribution_observed.sum()
    
    test_statistics, p_value = scipy.stats.chisquare(distribution_observed, distribution_expected)
    return p_value
    
