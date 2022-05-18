from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from concept_drift import statistical_tests
import helper # TODO move used helper functions to this module

class AttributeImportanceMeasurer():
    def __init__(self, window_creator, start_before_change_point, difference_measure):
        """Initiate a standard attribute importance measurer.
        
        Args:
            window_creator: A WindowCreator.
            start_before_change_point: How many observation before the change_point to take into account measuring attribute importance.
            difference_measure: How the attribute importance is measured. Instance of DifferenceMeasure.
        """
        self.window_creator = window_creator
        self.start_before_change_point = start_before_change_point
        self.difference_measure = difference_measure
        
    def get_attribute_importance(self, event_log, change_point_location):
        """Get the attribute importance around specific change_point
        
        Args:
            event_log: Event log for which change point is known and attributes should be explained.
            change_point_location: Location of the change point in the event log.
        
        Returns:
            Series with importance measure results
        """
        windows_start = change_point_location - self.start_before_change_point
        windows_end = change_point_location + self.window_creator.window_size
        
        # print(windows_start)
        # print(windows_end)
        
        # get log windows
        log_windows = self.window_creator.create_windows(event_log, windows_start, windows_end)
        
        # print(len(log_windows))
        # print()
        
        # calculate the importance measure
        attribute_importance_scores = self.difference_measure.calculate(log_windows)
        return attribute_importance_scores

class DifferenceMeasure(ABC):
    @abstractmethod
    def _test(self, data_series_a, data_series_b):
        pass
    
    @abstractmethod
    def _aggregate(self, test_results_df):
        pass
    
    def calculate(self, log_windows):
        """Calculate a difference measure for all windows in log.
        """
        calculation_results = {}
        for start, (window_a, window_b) in log_windows.items():
            calculation_results[start] = {}

            attributes_window_a = helper.get_trace_attributes(window_a)
            attributes_window_b = helper.get_trace_attributes(window_b)

            # perform test for each attribute
            # TODO handle the case that an attribute is not present in a given window
            for attribute_name, data_series_a in attributes_window_a.items():
                data_series_b = attributes_window_b[attribute_name]
                # perform the test
                test_result = self._test(data_series_a, data_series_b)
                calculation_results[start][attribute_name] = test_result
        
        # aggregate the test results
        results_df = pd.DataFrame().from_dict(calculation_results, orient='index')
        aggregated_results = self._aggregate(results_df)
        
        return aggregated_results

class DifferenceMeasureHellinger(DifferenceMeasure):
    def _get_probabilities(self, data_series_a, data_series_b):
         # get value counts for each series
        value_counts_a = data_series_a.value_counts()
        value_counts_b = data_series_b.value_counts()
        
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
    
    def _test(self, data_series_a, data_series_b):
        # transform data series into probabilities
        probas_a, probas_b = self._get_probabilities(data_series_a, data_series_b)
        hellinger_distance = np.sqrt(np.sum((np.sqrt(probas_a) - np.sqrt(probas_b)) ** 2)) / np.sqrt(2)
        return hellinger_distance
    
    def _aggregate(self, test_results_df):
        """Return the maximal returned distance."""
        return test_results_df.max()
    
class DifferenceMeasureChiSquare(DifferenceMeasure):
    def __init__(self, threshold):
        self.threshold = threshold
    
    def _test(self, data_series_a, data_series_b):
        return statistical_tests.test_chi_squared(data_series_a, data_series_b)
    
    def _aggregate(self, test_results_df):
        return self._get_consecutive_under_threshold(test_results_df, self.threshold)
    
    def _get_consecutive_trues(self, df):
        """Gets the number of consecutive True values in each column of the dataframe.

        See https://stackoverflow.com/a/52718619.

        Args:
            df: Dataframe with only True and False values.

        Returns:
            Series with dataframe column names as index and consecutive True value counts as values.
        """
        # get the number of consecutive True values per Column
        b = df.cumsum()
        c = b.sub(b.mask(df).ffill().fillna(0)).astype(int)

        mask = df.any()
        consecutive_trues_array = np.where(mask, c.max(), -1)

        # put result into pandas Series
        consecutive_trues_series = pd.Series(consecutive_trues_array, index=df.columns)
        
        # change all results that are negative values to 0
        consecutive_trues_series[consecutive_trues_series < 0] = 0

        return consecutive_trues_series

    def _get_consecutive_under_threshold(self, df, threshold=0.05):
        """Counts the number of consecutive times that values in a column of a dataframe are smaller than a threshold.

        Args:
            df: Dataframe for which to count consecutive values under threshold.

        Returns:
            Series with dataframe column names as index and consecutive True value counts as values.
        """
        # convert the dataframe into a boolean data frame
        boolean_df = df < threshold

        # evaluate the number of consecutive True values per column
        consecutive_trues_series = self._get_consecutive_trues(boolean_df)

        return consecutive_trues_series