"""Module for the detection of drift. Can be used for either the primary or secondary change perspective.
"""

import pandas as pd

from processdrift.framework import feature_extraction

class DriftDetector:
    """The drift detector gets a feature's change over time and can retrieve the according change points.
    """
    
    def __init__(self, feature_extractor, window_generator, population_comparer, threshold=0.05, detector_name=None):
        """Create a new drift detector and supply strategies for feature extraction, window generation and population comparison.
        
        Args:
            feature_extractor: A feature extractor from processdrift.framework.feature_extraction.
            window_generator: A window generator from processdrift.framework.windowing.
            population_comparer: A population comparer from processdrift.framework.popcomparison.
            threshold: The threshold in comparison value which identifies a change point.
            detector_name: The name of the detector.
        """
        self.feature_extractor = feature_extractor
        self.window_generator = window_generator
        self.population_comparer = population_comparer
        self.threshold = threshold
        self._name = detector_name
    
    @property
    def name(self):
        # The detector name is taken from the feature extractor if not otherwise specified
        if self._name == None:
            return self.feature_extractor.name
        else:
            return self._name
    
    def get_change_series(self, event_log):
        """Get the change over time from the event log.
        
        Args:
            event_log: A pm4py event log.
            
        Returns:
            The comparison result over time.
        """
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
        """Get the change points for a given event log using the defined threshold.
        
        Args:
            event_log: A pm4py event log.
            
        Returns:
            List of change points.
        """
        change_series = self.get_change_series(event_log)
        
        change_points = change_series[change_series > self.threshold].index
        
        return change_points


def get_all_attribute_drift_detectors(log, window_generator, population_comparer, threshold=0.05, exclude_attributes=[]):
    """Factory function to get attribute drift detectors for all trace level attributes in an event log.
    
    TODO implement for event level attributes as well.
    
    Args:
        log: A pm4py event log.
        window_generator: A windowing.WindowGenerator() to know which windowing strategy to use.
        pupulation_comparer: A pop_comparison.PopComparer() to know how to compare the populations.
        threshold: The threshold for change detection.
        exclude_attributes: Event log attributes for which no drift detector should be generated.
    
    Returns:
        List of drift detectors, one for each attribute.
    """
    
    # get all trace attributes
    trace_attributes = feature_extraction.get_all_trace_attributes(log)
    
    # remove attributes that shall be excluded
    trace_attributes.difference_update(exclude_attributes)
    
    # create the new feature extractors and detectors
    drift_detectors = []
    for attribute_name in trace_attributes:
        # get the unique feature extractor
        new_feature_extractor = feature_extraction.AttributeFeatureExtractor(attribute_level='trace', attribute_name=attribute_name)
        
        # create the drift detector
        drift_detector = DriftDetector(new_feature_extractor, window_generator, population_comparer, threshold=threshold)
        drift_detectors.append(drift_detector)
    
    return drift_detectors
