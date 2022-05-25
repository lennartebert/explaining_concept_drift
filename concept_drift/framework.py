"""Implementation of the concept drift explanation framework.
"""
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from concept_drift import features
from concept_drift import statistical_tests
from concept_drift import windows

class DriftExplainer():
    """The concept drift explainer explains concept drift which is observable in an event log by ranking attributes by their importance.
    """
    
    def __init__(self, primary_drift_detector, secondary_drift_detectors):
        self.primary_drift_detector = primary_drift_detector
        self.secondary_drift_detectors = secondary_drift_detectors
        
    def get_primary_and_secondary_change_series(self, event_log):
        primary_change_series = self.primary_drift_detector.get_change_series(event_log)
        secondary_change_series_dict = {}
        for secondary_drift_detector in self.secondary_drift_detectors:
            secondary_change_points = secondary_drift_detector.get_change_series(event_log)
            secondary_change_series_dict[secondary_drift_detector.detector_name] = secondary_change_points
        
        return primary_change_series, secondary_change_series_dict
    
    def attribute_importance_per_primary_change_point(self, event_log):
        # get process concept drift points
        primary_change_points = self.primary_drift_detector.get_change_points(event_log)
        
        # get secondary drifts
        # the resulting dictionary will have the format
        # {detector_name: change_point_list}, e.g., {'attribute XXX': [492, 2849]}
        
        all_secondary_change_points = {}
        for secondary_drift_detector in self.secondary_drift_detectors:
            secondary_change_points = secondary_drift_detector.get_change_points(event_log)
            all_secondary_change_points[secondary_drift_detector.detector_name] = secondary_change_points
            
        # rank attributes by closest secondary change point before primary change point
        # for that, create a list of tuples with all secondary change points and detector names
        secondary_time_detector_tuples = []
        for drift_detector, change_points in all_secondary_change_points.items():
            for change_point in change_points:
                secondary_time_detector_tuples.append((change_point, drift_detector))
        
        # get rank the secondary change point detectors by how close the secondary change point was to the primary change point
        change_point_explanations = {}
        for primary_change_point in primary_change_points:            
            distances_to_change_point = []
            for secondary_change_point, drift_detector in secondary_time_detector_tuples:
                # only search up to the primary change point
                if primary_change_point > secondary_change_point:
                    break
                
                distance = primary_change_point - secondary_change_point
                
                distances_to_change_point.append({
                    'detector': drift_detector,
                    'detector_change_point': secondary_change_point,
                    'distance': distance
                })
                
            # revert sorting of change point distance list so that the closest observed change point comes first
            distances_to_change_point.reverse()
            
            change_point_explanations[primary_change_point] = distances_to_change_point
            
        return change_point_explanations

class DriftDetector:
    def __init__(self, feature_extractor, window_generator, population_comparer, threshold=0.05, detector_name=None):
        self.feature_extractor = feature_extractor
        self.window_generator = window_generator
        self.population_comparer = population_comparer
        self.threshold = threshold
        self._detector_name = detector_name
    
    @property
    def detector_name(self):
        if self._detector_name == None:
            return self.feature_extractor.name
        else:
            return self._detector_name
    
    def get_change_series(self, event_log):
        change_dictionary = {}
        
        # get windows for comparison
        for window_a, window_b in self.window_generator.get_windows(event_log):
            # get features for each window
            features_window_a = self.feature_extractor.extract(window_a.log)
            features_window_b = self.feature_extractor.extract(window_b.log)
            
            # print(pd.concat([features_window_a, features_window_b], axis=1))# , right_index=True, left_index=True))
            
            # compare both windows
            # result is true if a significant change is found
            result = self.population_comparer.compare(features_window_a, features_window_b)
            
            change_dictionary[window_b.start] = result
        
        change_series = pd.Series(change_dictionary)
        return change_series
    
    def get_change_points(self, event_log):
        """Get the change points for a given event log."""
        change_series = self.get_change_series(event_log)
        
        change_points = change_series[change_series > self.threshold].index
        
        return change_points
        

class WindowGenerator(ABC):
    @abstractmethod
    def get_windows(self, event_log, start=None, end=None):
        pass

class FixedSizeWindowGenerator(WindowGenerator):
    def __init__(self,  window_size, window_offset=None, slide_by=None, start=None, end=None, inclusion_criteria='events'):
        """Initialize the window generator with the desired settings.
        
        Args:
            window_size: Size of each window as Python datetime.timedelta or trace number.
            window_offset: Offset of windows defined as datetime.timedelta or trace number. If None, offsets by window_size (non-overlapping windows).
            slide_by: How much to slide between generated windows. Defaults to window_size.
            inclusion_criteria: 'events', 'traces_intersecting', 'trace_contained'. Either return all events that take place in a window, all complete traces that have any event in the window or all complete traces that are fully contained in the window. Ignored if type is 'traces'.
        """
        self.window_size = window_size
        self.window_offset = window_offset
        self.slide_by = slide_by
        self.inclusion_criteria = inclusion_criteria
        
    def get_windows(self, event_log, start=None, end=None):
        """Create windows of the given log according to the initialization parameters.
        
        Args:
            event_log: An event log to create the windows for.
            start: Optional argument. If the start of the windowing should not be the first event in the log.
            end: Optional argument. To be set if the end of the windowing should not be the last complete window.
        
        Returns:
            Yields list of windows [(window_a, window_b), ...].
        """
        for got_windows in windows.get_log_windows(event_log, self.window_size, self.window_offset, self.slide_by, start, end, self.inclusion_criteria):
            yield got_windows
    
# TODO implement AdaptiveWindowGenerator

class FeatureExtractor(ABC):    
    @abstractmethod
    def extract(self, log):
        pass
    
    @property
    @abstractmethod
    def name(self):
        pass
    
    def __repr__(self):
        return self.name

def get_all_trace_attributes(log, exclude_attributes=[]):
    attribute_set = set()
    for trace in log:
        attribute_set.update(trace.attributes)
    
    # remove attributes that shall be excluded
    attribute_set.difference_update(exclude_attributes)
    
    return attribute_set


# factory function to get all attribute feature extractors
# TODO only implemented for trace level at this point
def get_all_attribute_drift_detectors(log, window_generator, population_comparer, threshold=0.05, exclude_attributes=[]):
    trace_attributes = get_all_trace_attributes(log, exclude_attributes=exclude_attributes)
    
     # create the new feature extractors and explainers
    drift_detectors = []
    for attribute_name in trace_attributes:
        # get the unique feature extractor
        new_feature_extractor = AttributeFeatureExtractor(attribute_level='trace', attribute_name=attribute_name)
        
        # create the drift detector
        drift_detector = DriftDetector(new_feature_extractor, window_generator, population_comparer, threshold=threshold)
        drift_detectors.append(drift_detector)
    
    return drift_detectors
    
class AttributeFeatureExtractor(FeatureExtractor):
    def __init__(self, attribute_level, attribute_name):
        self.attribute_level = attribute_level
        self.attribute_name = attribute_name
        
    def extract(self, log):
        result_list = []
        
        if self.attribute_level == 'trace':
            for trace in log:
                result_list.append(trace.attributes[self.attribute_name])
        
        # convert to numpy array
        result_array = np.array(result_list)
        
        return result_array
    
    @property
    def name(self):
        return self.attribute_name
    

class RelationalEntropyFeatureExtractor(FeatureExtractor):
    def __init__(self, direction='followed_by', activity_name_field='concept:name'):
        self.direction = direction
        self.activity_name_field = activity_name_field
    
    def extract(self, log):
        return features.get_relational_entropy(log, direction=self.direction, activity_name_field=self.activity_name_field)
    
    @property
    def name(self):
        return 'Relational Entropy'

class PopulationComparer(ABC):
    def _preprocess(self, population_1, population_2):
        return population_1, population_2
    
    @abstractmethod
    def _get_comparison_measure(self, population_1, population_2):
        pass
    
    def compare(self, population_1, population_2):
        population_1_preprocessed, population_2_preprocessed = self._preprocess(population_1, population_2)
        difference = self._get_comparison_measure(population_1_preprocessed, population_2_preprocessed)
        return difference
    
class KSTestPopulationComparer(PopulationComparer):
    def _get_comparison_measure(self, population_1, population_2):
        return statistical_tests.test_kolmogorov_smirnov(population_1, population_2)

class ProbabilityDistributionComparer(PopulationComparer, ABC):
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
    def _get_comparison_measure(self, population_1, population_2):
        hellinger_distance = np.sqrt(np.sum((np.sqrt(population_1) - np.sqrt(population_2)) ** 2)) / np.sqrt(2)
        return hellinger_distance
    
class ProbabilitisticPopulationComparer(PopulationComparer):
    
    
    def _test(self, data_series_a, data_series_b):
        # transform data series into probabilities
        probas_a, probas_b = self._get_probabilities(data_series_a, data_series_b)
        hellinger_distance = np.sqrt(np.sum((np.sqrt(probas_a) - np.sqrt(probas_b)) ** 2)) / np.sqrt(2)
        return hellinger_distance
    


    