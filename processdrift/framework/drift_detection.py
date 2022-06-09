"""Module for the detection of drift. Can be used for either the primary or secondary change perspective.
"""

import pandas as pd
import numpy as np
import subprocess

from processdrift.framework import feature_extraction
from processdrift.framework import windowing
import pm4py

class DriftDetector:
    """The drift detector gets a feature's change over time and can retrieve the according change points.
    """
    
    def __init__(self, feature_extractor, window_generator, population_comparer, threshold=0.05, min_observations_below=3, min_distance_change_streaks=3, detector_name=None):
        """Create a new drift detector and supply strategies for feature extraction, window generation and population comparison.
        
        Args:
            feature_extractor: A feature extractor from processdrift.framework.feature_extraction.
            window_generator: A window generator from processdrift.framework.windowing.
            population_comparer: A population comparer from processdrift.framework.popcomparison.
            threshold: The threshold in comparison value which identifies a change point.
            min_observations_below: Minimum number of observations below the threshold for a change point to be registered.
            min_distance_change_streaks: Minimum distance between two change points.
            detector_name: The name of the detector.
        """
        self.feature_extractor = feature_extractor
        self.window_generator = window_generator
        self.population_comparer = population_comparer
        self.threshold = threshold
        self.min_observations_below = min_observations_below
        self.min_distance_change_streaks = min_distance_change_streaks
        self._name = detector_name
    
    @property
    def name(self):
        # The detector name is taken from the feature extractor if not otherwise specified
        if self._name == None:
            return self.feature_extractor.name
        else:
            return self._name

    def get_changes(self, event_log, around_traces=None, max_distance=None):
        """Get changes in the selected feature from an event log.

        The search for changes can be restricted to the area of max_distance around traces specified in a list.

        Args:
            event_log: A pm4py event log.
            around_traces: List of trace numbers. Only look at changes around traces.
            max_distance: Maximum distance around each trace to look for a change.

        Returns:
            Dictionary with change points and change series: {change_points: [...], change_series: pandas.Series}
        """

        # get the change series
        change_series = self._get_change_series(event_log, around_traces=around_traces, max_distance=max_distance)

        # get the change points
        change_points = self._get_change_points(change_series)

        result = {
            'change_points': change_points,
            'change_series': change_series
        }

        return result

    def _get_change_series(self, event_log, around_traces, max_distance):
        """Get the change over time from the event log.
        
        Args:
            event_log: A pm4py event log.
            around_traces: List of trace numbers. Only look at changes around traces.
            max_distance: Maximum distance around each trace to look for a change.
            
        Returns:
            The comparison result over time.
        """
        change_dictionary = {}

        # make sure that around_traces is sorted
        around_traces.sort()
        
        # get windows for comparison
        for window_a, window_b in self.window_generator.get_windows(event_log):
            # check if around_traces is set, if yes, then continue to next window if this one is not in scope
            is_around_trace = False
            if around_traces is not None:
                window_location = window_b.start
                for trace_location in around_traces:
                    if trace_location > window_location + self.window_generator.window_size:
                        break
                    elif abs(trace_location - window_location) <= max_distance:
                        is_around_trace = True
            
            if not is_around_trace:
                change_dictionary[window_b.start] = None
                continue

            # get features for each window
            features_window_a = self.feature_extractor.extract(window_a.log)
            features_window_b = self.feature_extractor.extract(window_b.log)

            if len(features_window_a) != len(features_window_b):
                print('lenghts of windows does not match.')
            
            # update window size
            if isinstance(self.window_generator, windowing.AdaptiveWindowGenerator):
                self.window_generator.update_window_size(features_window_a, features_window_b)
            
            # print(pd.concat([features_window_a, features_window_b], axis=1))# , right_index=True, left_index=True))
            
            # compare both windows
            # result is true if a significant change is found
            result = self.population_comparer.compare(features_window_a, features_window_b)
            
            change_dictionary[window_b.start] = result
        
        change_series = pd.Series(change_dictionary)
        return change_series


    # def _get_change_points(self, event_log):
    #     """Get the change points for a given event log using the defined threshold.
        
    #     Args:
    #         event_log: A pm4py event log.
        
    #     Returns:
    #         List of change points.
    #     """
    #     change_series = self.get_change_series(event_log)
        
    #     change_points = self.get_change_points_from_series(change_series)
        
    #     return change_points
    

    def _get_change_points(self, series):
        """Gets a list of changepoints from a series of observations based on a threshold. 
        
        A change point is registered if the series values are below the threshold for the duration of 'min_observations_below'. 
        Change points will only be flagged if there is at least a distance of 'min_distance_change_streaks' between streaks of observations below the change point.
        
        Args:
            series: Pandas series with observations.
            
        Returns:
            List of change points.
        """
        series_below_threshold = series <= self.threshold
        below_threshold_counts = series_below_threshold * (series_below_threshold.groupby((series_below_threshold != series_below_threshold.shift()).cumsum()).cumcount() + 1)

        change_points = []
        last_change_streak_ended = None
        for index, streak_count in below_threshold_counts.iteritems():  
            # check if the streak is exactly the minimum number of observations
            if streak_count == self.min_observations_below:
                
                change_point_candidate = index - self.min_observations_below + 1

                # definitely enter the change point if this is the first streak that was seen
                if last_change_streak_ended is None:
                    change_points.append(change_point_candidate)
                else:
                    # get distance to last streak end
                    distance_to_last_streak = change_point_candidate - last_change_streak_ended - 1

                    if distance_to_last_streak >= self.min_distance_change_streaks:
                        change_points.append(change_point_candidate)

            # populate the last change streak if the count is not 0
            if streak_count >= self.min_observations_below:
                last_change_streak_ended = index

        return change_points


class DriftDetectorProDrift(DriftDetector):
    """Leverages ProDrift for drift detection purposes."""
    def __init__(self, 
                    path_to_prodrift, 
                    drift_detection_mechanism='runs',
                    window_size=200,
                    window_mode='adaptive',
                    detect_gradual_as_well=False):
        self.path_to_prodrift = path_to_prodrift
        self.drift_detection_mechanism = drift_detection_mechanism
        self.window_size = window_size
        self.window_mode = window_mode
        self.detect_gradual_as_well = detect_gradual_as_well

    def get_changes(self, event_log):
        """Get changes in the selected feature from an event log.

        The search for changes can be restricted to the area of max_distance around traces specified in a list.

        Args:
            event_log: A pm4py event log.

        Returns:
            Dictionary with change points and change series: {change_points: [...], change_series: pandas.Series}
        """
 
        # get the change points
        change_points = self._get_change_points(event_log)

        # get the change series
        change_series = self._get_change_series(event_log, change_points)
        
        result = {
            'change_points': change_points,
            'change_series': change_series
        }

        return result

    def _get_change_series(self, event_log, change_points): # TODO diverging implementation from derived class!
        # For the ProDrift drift detector, we do not have the scalar value of the change measure.
        # Therefore, the change series is considered always 1, as long as there is no change point detected.

        # get the number of traces in the log
        number_traces = len(event_log)

        # create the numpy array with all 1 values
        change_series_array = np.ones((number_traces,), dtype=int)
        
        # replace each one with a 0 at the change point
        for change_point in change_points:
            change_series_array[change_point] = 0
        
        # get as pandas series
        change_series = pd.Series(change_series_array)

        return change_series

    # def get_change_points_from_series(self, series):
    #     """Gets a list of changepoints from a series of observations based on a threshold. 
        
    #     The ProDrift implementation will just return all the points in the change_series where the value == 0.

    #     Args:
    #         series: Pandas series with observations.
            
    #     Returns:
    #         List of change points.
    #     """

    #     indeces_change_points = series[series == 0].index

    #     # convert to list

    #     change_points = list(indeces_change_points)

    #     return change_points

    def _get_change_points(self, event_log):
        # save event log to temporary file
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmp:
            # save the event log in a temporary directory
            event_log_path = os.path.join(tmp, 'event_log.xes')
            pm4py.objects.log.exporter.xes.exporter.apply(event_log, event_log_path)

            # build the ProDrift command
            command = f"java -jar \"{self.path_to_prodrift}\" -fp \"{event_log_path}\"" \
                f" -ddm {self.drift_detection_mechanism}"
            
            if self.window_size is not None:
                command += f" -ws {self.window_size}"
            
            if self.window_mode is not None:
                if self.window_mode == 'adaptive':
                    pass # adaptive mode is the default
                else:
                    command += " -fwin"

            if self.detect_gradual_as_well:
                command += " -gradual"
            
            print(command)

            # run the command
            output = str(subprocess.check_output(command))
            print(output)

            # get the change points
            change_points = []
            for line in output.split('\\n'):
                # see if the output line contains a drift point
                if 'drift detected at trace: ' in line:
                    line_starting_with_trace_number = line.split('drift detected at trace: ')[1]
                    trace_number = line_starting_with_trace_number.split(' ')[0]
                    change_points.append(int(trace_number))

            return change_points
    

class DriftDetectorTrueKnown(DriftDetector):
    """A drift detector to be used if the true change points are known.

    Will always return 100% accurate results
    """
    def __init__(self, change_points):
        """Create a new drift detector and supply strategies for feature extraction, window generation and population comparison.
        
        Args:
            change_points: Dictionary with drift points
        """
        self.change_points = change_points
        self.feature_extractor = None
        self.window_generator = None
        self.population_comparer = None
        self.threshold = None
        self._name = 'Drift Detector True Known'
    
    def get_changes(self, event_log):
        result = {
            'change_points': self.change_points,
            'change_series': self._get_change_series(event_log)
        }

        return result
    
    def _get_change_series(self, event_log):
        """Get the change over time from the known drift points.
        
        Args:
            event_log: A pm4py event log.
            
        Returns:
            The comparison result over time.
        """

        # start by populating the series with 0 values
        number_traces = len(event_log)
        change_series = pd.Series(data=[1] * number_traces, index=range(number_traces))

        # now insert the changepoints
        change_series.loc[self.change_points] = 0

        return change_series
    
    def _get_change_points(self, event_log):
        """Get the change points as presented when initialized.
        
        Args:
            event_log: A pm4py event log.
            
        Returns:
            List of change points.
        """
        return self.change_points

def get_all_attribute_drift_detectors(log, window_generator, population_comparer, threshold=0.05, exclude_attributes=[], min_observations_below=3, min_distance_change_streaks=3):
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
        drift_detector = DriftDetector(new_feature_extractor, window_generator, population_comparer, threshold=threshold, min_observations_below=min_observations_below, min_distance_change_streaks=min_observations_below)
        drift_detectors.append(drift_detector)
    
    return drift_detectors

