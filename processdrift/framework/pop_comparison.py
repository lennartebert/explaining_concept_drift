"""Module for the comparison of populations in the process mining concept drift explanation framework.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import scipy.stats
import math

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

    def __repr__(self):
        return self.__class__.__name__
    
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


class HotellingsTSquaredPopComparer(PopComparer):
    """Perform a Hotellings T Squared test to compare two populations.
    """
    def _get_comparison_measure(self, population_1, population_2):
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

def get_contingency_table(pop_1_array, pop_2_array):
    """Get a numpy contingency table from two arrays of samples.
    
    Args:
        pd_series_1: First sample array.
        pd_series_2: Second sample array.
    """
    # get pandas series for both arrays of observations so that we can calculate the value counts next
    pop_1 = pd.Series(pop_1_array)
    pop_2 = pd.Series(pop_2_array)

    # get the observation frequencies
    pop_1_frequencies = pop_1.value_counts()
    pop_2_frequencies = pop_2.value_counts()

    # create the contingency table, fill in NaN with 0
    contingency_table_df = pd.DataFrame([pop_1_frequencies, pop_2_frequencies])
    contingency_table_df = contingency_table_df.fillna(0)

    # convert to numpy array
    contingency_table = contingency_table_df.to_numpy()
    return contingency_table

class ChiSquaredComparer(PopComparer):
    """Get the Chi Squared Distance between the two populations.
    """    
    def _get_comparison_measure(self, population_1, population_2):

        """Calculates the Chi-square test between the two populations.
        
        Args:
            population_1: The first population.
            population_2: The first population.
            
        Returns:
            p-value for Chi-Square test for both populations.
        """
        # get the contingency table
        contingency_table = get_contingency_table(population_1, population_2)

        # calculate the chi-squared p-value
        stat, p_value, dof, expected = scipy.stats.chi2_contingency(contingency_table)

        # check if all expected values are >= 5
        if (expected < 5).any():
            p_value = np.NaN

        return p_value

class GTestComparer(PopComparer):
    """Get the G-test result for the two populations.
    """
    def _get_comparison_measure(self, population_1, population_2):
        """Calculates the G-test between the two populations.
        
        Args:
            population_1: The first population.
            population_2: The first population.
            
        Returns:
            p-value for G-test for both populations.
        """
        # get the contingency table
        contingency_table = get_contingency_table(population_1, population_2)

        # calculate the chi-squared p-value
        stat, p_value, dof, expected = scipy.stats.chi2_contingency(contingency_table, lambda_="log-likelihood")

        # check if all expected values are >= 5
        if (expected < 5).any():
            p_value = np.NaN
        
        return p_value


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

