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


def get_frequency_counts(observations):
    """Get frequency counts from a list of observations.
    
    Args:
        observations: List of observations
        
    Returns:
        Series with counts for each observation.
    """
    observations_series = pd.Series(observations)
    frequency_counts = observations_series.value_counts()
    return frequency_counts


class ChiSquaredComparer(PopComparer):
    """Get the Chi Squared Distance between the two populations.
    """
    def _preprocess(self, population_1, population_2):
        """Get frequency counts for both populations.

        Args:
            population_1: The first population.
            population_2: The second population.
        
        Returns:
            (population_1, population_2): The two preprocessed populations.
        """
        # create series from observation list
        pop_1_series = pd.Series(population_1, name='expected')
        pop_2_series = pd.Series(population_2, name='observed')

        # count number of values
        pop_1_value_counts = pop_1_series.value_counts()
        pop_2_value_counts = pop_2_series.value_counts()

        # create pandas dataframe
        merged_df = pd.merge(pop_1_value_counts, pop_2_value_counts, how='left', left_index=True, right_index=True)

        # replace Na with 0
        merged_df = merged_df.fillna(0)

        # set data type to integer
        merged_df = merged_df.astype('int')

        # sum the expected values and oversample the observed values to fit
        missing_observed_samples = sum(merged_df['expected']) - sum(merged_df['observed'])
        missing_observed_samples

        # draw the sample
        p = merged_df['observed'] / sum(merged_df['observed'])
        
        # make sure p is never NaN
        p = p.fillna(0)
        
        oversampled_observations = np.random.choice(list(merged_df.index), size=missing_observed_samples, p=p)

        # convert to pandas series
        oversampled_series = pd.Series(oversampled_observations)
        oversampled_value_counts = oversampled_series.value_counts()

        # add to observed axis 
        merged_df['observed'] = merged_df['observed'].add(oversampled_value_counts, fill_value=0)

        # set data type to integer
        merged_df = merged_df.astype('int')

        return merged_df['expected'].values, merged_df['observed'].values
    
    def _get_comparison_measure(self, population_1, population_2):
        """Calculates the Chi-square test between the two populations.
        
        Args:
            population_1: The first population. Also called the expected population.
            population_2: The first population. Also called the observed population.
            
        Returns:
            p-value for Chi-Square test for both populations.
        """
        # the populations have been transformed into frequency counts in preprocessing

        test_statistics, p_value = scipy.stats.chisquare(population_2, population_1)
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

