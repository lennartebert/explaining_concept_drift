"""Module for the comparison of populations in the process mining concept drift explanation framework.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import scipy

import spm1d

class PopComparer(ABC):
    """PopComparer compares two populations.
    
    The lower the resulting value, the more significantly different are both populations.
    """
    
    def _preprocess(self, population_1, population_2):
        """Preprocess both populations. Does not need to be implemented.
        
        Note that both populations are passed in case the preprocessing needs to know both populations, e.g., for min-max scaling.
        
        Args:
            population_1: The first population.
            population_2: The second population.
            
        Returns:
            (population_1, population_2): The two preprocessed populations.
        """
        return population_1, population_2
    
    @abstractmethod
    def _get_comparison_measure(self, population_1, population_2):
        """Calculates the comparison measure between two populations. The lower, the more significantly different are the two populations.
        
        Args:
            population_1: The first population. Also called the expected population.
            population_2: The first population. Also called the observed population.
            
        Returns:
            Comparison measure between 0 and 1.
        """
        pass
    
    def compare(self, population_1, population_2):
        """Preprocesses and compares two populations
        
        Args:
            population_1: The first population. Also called the expected population.
            population_2: The first population. Also called the observed population.
            
        Returns:
            Comparison measure between 0 and 1.
        """
        population_1_preprocessed, population_2_preprocessed = self._preprocess(population_1, population_2)
        difference = self._get_comparison_measure(population_1_preprocessed, population_2_preprocessed)
        return difference
    
class KSTestPopComparer(PopComparer):
    """Perform a two-sample Kolmogorov-Smirnov test to compare two populations.
    """
    def _get_comparison_measure(self, population_1, population_2):
        """Calculates the Kolmogorov-Smirnov test as comparison measure measure between the two populations.
        
        Args:
            population_1: The first population. Also called the expected population.
            population_2: The first population. Also called the observed population.
            
        Returns:
            P-value of KS test.
        """
        test_statistics, p_value = scipy.stats.kstest(population_1, population_2)
        return p_value

class ProbabilityDistributionComparer(PopComparer, ABC):
    """Abstract class to be inhereted from if the PopComparer works based on probability distributions. 
    """
    def _preprocess(self, population_1, population_2):
        pop_1_series = pd.Series(population_1)
        pop_2_series = pd.Series(population_2)
        
         # get value counts for each series
        value_counts_a = pop_1_series.value_counts()
        value_counts_b = pop_2_series.value_counts()
        
        # get the relative probabilities
        probas_a = value_counts_a / sum(value_counts_a)
        probas_b = value_counts_b / sum(value_counts_b)
        
        # let both arrays have the same length
        value_counts_df = pd.concat([probas_a, probas_b], axis=1)

        # rename the columns
        value_counts_df.columns = ['expected', 'observed']

        # replace nan with 0
        value_counts_df = value_counts_df.fillna(0)

        # compute the observed frequencies in each category
        distribution_expected = value_counts_df['expected']
        distribution_observed = value_counts_df['observed']
        
        return distribution_expected, distribution_observed

class HellingerDistanceComparer(ProbabilityDistributionComparer):
    """Get the Hellinger Distance between the two populations.
    
    In preprocessing, both populations are converted to probability distributions. 
    """
    
    def _get_comparison_measure(self, population_1, population_2):
        """Calculates the Hellinger Distance between the two populations.
        
        Args:
            population_1: The first population. Also called the expected population.
            population_2: The first population. Also called the observed population.
            
        Returns:
            Hellinger Distance (0 to 1, the lower, the more different).
        """
        hellinger_distance = np.sqrt(np.sum((np.sqrt(population_1) - np.sqrt(population_2)) ** 2)) / np.sqrt(2)
        
        # we return 0 when both populations are different and 1 if they are the same. Therefore, the metric needs to be inverted.
        comparison_measure = 1 - hellinger_distance
        return comparison_measure

# TODO finalize docstrings and delete unused functions.
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
    
