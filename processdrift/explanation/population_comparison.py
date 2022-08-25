"""Module for the comparison of populations in the process mining concept drift explanation explanation.
"""
from collections import Counter
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import scipy.stats
import math

class PopulationComparer(ABC):
    """PopulationComparer compares two populations.
    
    The lower the resulting value, the more significantly different are both populations.
    """
    
    @abstractmethod
    def compare(self, population_1, population_2):
        """Preprocesses and compares two populations.
        
        Args:
            population_1: The first population. Also called the expected population.
            population_2: The first population. Also called the observed population.
            
        Returns:
            Comparison measure between 0 and 1.
        """
        pass

    def __repr__(self):
        return self.__class__.__name__

class KSTestPC(PopulationComparer):
    """Perform a two-sample Kolmogorov-Smirnov test to compare two populations.
    """
    def compare(self, population_1, population_2):
        """Calculates the Kolmogorov-Smirnov test as comparison measure measure between the two populations.
        
        Args:
            population_1: The first population. Also called the expected population.
            population_2: The first population. Also called the observed population.
            
        Returns:
            P-value of KS test.
        """
        # The populations can hold None values. These need to be filtered out.
        pop_1_no_nan = [sample for sample in population_1 if sample is not None]
        pop_2_no_nan = [sample for sample in population_2 if sample is not None]

        # return None if either population 1 or population 2 has no samples that are not None
        if len(pop_1_no_nan) == 0 or len(pop_2_no_nan) == 0:
            return None

        test_statistics, p_value = scipy.stats.kstest(pop_1_no_nan, pop_2_no_nan)
        return p_value


class HotellingsTSquaredPC(PopulationComparer):
    """Perform a Hotellings T Squared test to compare two populations.
    """
    def compare(self, population_1, population_2):
        """Calculates the Hotellings T Squared test as comparison measure measure between the two populations.
        
        Args:
            population_1: The first population. Also called the expected population.
            population_2: The first population. Also called the observed population.
            
        Returns:
            P-value of Hotellings T Squared test.
        """
        # the test cannot be calculated if the populations are singular
        
        p_value = None
        
        try:
            # implementation by https://www.r-bloggers.com/2020/10/hotellings-t2-in-julia-python-and-r/
            X = population_1.to_numpy()
            Y = population_2.to_numpy()
            
            nx, p = X.shape
            ny, _ = Y.shape
            
            delta = np.mean(X, axis=0) - np.mean(Y, axis=0)

            Sx = np.cov(X, rowvar=False)
            Sy = np.cov(Y, rowvar=False)

            S_pooled = ((nx-1)*Sx + (ny-1)*Sy)/(nx+ny-2)
            t_squared = (nx*ny)/(nx+ny) * np.matmul(np.matmul(delta.transpose(), np.linalg.inv(S_pooled)), delta)
            statistic = t_squared * (nx+ny-p-1)/(p*(nx+ny-2))

            f = scipy.stats.f(p, nx+ny-p-1)
            p_value = 1 - f.cdf(statistic)
        except np.linalg.LinAlgError as err:
             # catch linear algebra errors that are thrown when the populations are singular
            if 'Singular matrix' in str(err):
                p_value = 1
            else:
                raise err
        
        return p_value

class HellingerDistancePC(PopulationComparer):
    """Get the Hellinger Distance between the two populations.
    
    In preprocessing, both populations are converted to probability distributions. 
    """
    
    def compare(self, population_1, population_2):
        """Calculates the Hellinger Distance between the two populations.
        
        Args:
            population_1: The first population. Also called the expected population.
            population_2: The first population. Also called the observed population.
            
        Returns:
            Hellinger Distance (0 to 1, the lower, the more different).
        """
        p, q = preprocess_get_normalized_contingency_table(population_1, population_2)

        hellinger_distance = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)
        
        # we return 0 when both populations are different and 1 if they are the same. Therefore, the metric needs to be inverted.
        comparison_measure = 1 - hellinger_distance
        return comparison_measure

class ChiSquaredPC(PopulationComparer):
    """Get the Chi Squared Distance between the two populations.
    """

    def __init__(self, minimum_expected_frequency = 5):
        self.minimum_expected_frequency = minimum_expected_frequency


    def compare(self, population_1, population_2):

        """Calculates the Chi-square test between the two populations.
        
        Args:
            population_1: The first population.
            population_2: The first population.
            
        Returns:
            p-value for Chi-Square test for both populations.
        """
        # get the contingency table
        contingency_table = preprocess_get_contingency_table(population_1, population_2)

        # calculate the chi-squared p-value
        stat, p_value, dof, expected = scipy.stats.chi2_contingency(contingency_table)

        # check if all expected values are >= self.minimum_expected_frequency
        if (expected < self.minimum_expected_frequency).any():
            p_value = np.NaN

        return p_value

class GTestPC(PopulationComparer):
    """Get the G-test result for the two populations.
    """
    def __init__(self, minimum_expected_frequency = 5):
        self.minimum_expected_frequency = minimum_expected_frequency

    def compare(self, population_1, population_2):
        """Calculates the G-test between the two populations.
        
        Args:
            population_1: The first population.
            population_2: The first population.
            
        Returns:
            p-value for G-test for both populations.
        """
        # get the contingency table
        contingency_table = preprocess_get_contingency_table(population_1, population_2)

        # calculate the chi-squared p-value
        stat, p_value, dof, expected = scipy.stats.chi2_contingency(contingency_table, lambda_="log-likelihood")

        # check if all expected values are >= 5
        if (expected < self.minimum_expected_frequency).any():
            p_value = np.NaN
        
        return p_value

def preprocess_get_contingency_table(pop_1_array, pop_2_array):
    """Get a numpy contingency table from two arrays of samples.
    
    Args:
        pd_series_1: First sample array.
        pd_series_2: Second sample array.


    Returns:
        Contingency table as numpy array. Population 1 values in first row, population 2 values in second row.
    """
    pop_1_counter = Counter(pop_1_array)
    pop_2_counter = Counter(pop_2_array)
    result_array = []

    keys = list(pop_1_counter.keys() | pop_2_counter.keys())
    result_array = []

    for dict in [pop_1_counter, pop_2_counter]:
        result_array.append([dict.get(key, 0) for key in keys])

    return np.array(result_array)

def preprocess_get_normalized_contingency_table(pop_1_array, pop_2_array):
    """Gets a distribution table. This table is like a contingency table but normalizes the occurence frequncies.
        
    Args:
        pd_series_1: First sample array.
        pd_series_2: Second sample array.
    
    Returns:
        Normalized contingency table as numpy array. Population 1 values in first row, population 2 values in second row.
    """
    contingency_table = preprocess_get_contingency_table(pop_1_array, pop_2_array)
    sum_of_rows = contingency_table.sum(axis=1)
    normalized_array = contingency_table / sum_of_rows[:, np.newaxis]
    return normalized_array
