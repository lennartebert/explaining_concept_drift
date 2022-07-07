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

    def get_changes(self, event_log, around_change_points=None, max_distance=None):
        """Get changes in the selected feature from an event log.

        The search for changes can be restricted to the area of max_distance around traces specified in a list.

        Args:
            event_log: A pm4py event log.
            around_change_points: List of trace numbers. Only look at changes around traces.
            max_distance: Maximum distance around each trace to look for a change.

        Returns:
            Dictionary with change points and change series: {change_points: [...], change_series: pandas.Series}
        """

        # get the change series
        change_series = self._get_change_series(event_log, around_change_points=around_change_points, max_distance=max_distance)

        # get the change points
        change_points = self._get_change_points(change_series)

        result = {
            'change_points': change_points,
            'change_series': change_series
        }

        return result

    def _get_change_series(self, event_log, around_change_points, max_distance):
        """Get the change over time from the event log.
        
        Args:
            event_log: A pm4py event log.
            around_change_points: List of trace numbers. Only look at changes around traces.
            max_distance: Maximum distance around each trace to look for a change.
            
        Returns:
            The comparison result over time.
        """
        change_dictionary = {}

        # if the user specified change points, only search around these
        if around_change_points is not None:
            # make sure that around_change_points is sorted
            around_change_points.sort()

            last_end_change_point_window = None
            for change_point in around_change_points:
                # determine the area in which to look for changes
                start_change_point_window = max(0, change_point-max_distance)
                end_change_point_window = min(len(event_log), change_point+max_distance)

                # reset the size of the adaptive window generator if there was a gap between the last window areoun the change point and this one
                if last_end_change_point_window is not None:
                    # did the windows overlap -> set start_change_point_window to end of last change point window
                    if last_end_change_point_window > start_change_point_window:
                        start_change_point_window = last_end_change_point_window
                    # else, keep it as is and reset the adaptive window generator window size, if that window generator is used
                    else:
                        # update window size for adaptive generator
                        if isinstance(self.window_generator, windowing.AdaptiveWindowGenerator):
                            self.window_generator.reset_window_size()
                # update the last end of the change point window
                last_end_change_point_window = end_change_point_window
                
                window_generator_start = max(0, start_change_point_window-self.window_generator.window_size)

                # get windows for comparison
                for window_a, window_b in self.window_generator.get_windows(event_log, start=window_generator_start):
                    if window_b.start > end_change_point_window: break 

                    # get features for each window
                    features_window_a = self.feature_extractor.extract(window_a.log)
                    features_window_b = self.feature_extractor.extract(window_b.log)

                    # update window size for adaptive generator
                    if isinstance(self.window_generator, windowing.AdaptiveWindowGenerator):
                        self.window_generator.update_window_size(features_window_a, features_window_b)
                    
                    # compare both windows
                    # result is true if a significant change is found
                    result = self.population_comparer.compare(features_window_a, features_window_b)
                    
                    change_dictionary[window_b.end] = result
        # look for change globally, not just around change points
        else:
            # get windows for comparison
            for window_a, window_b in self.window_generator.get_windows(event_log):
                # get features for each window
                features_window_a = self.feature_extractor.extract(window_a.log)
                features_window_b = self.feature_extractor.extract(window_b.log)

                # update window size for adaptive generator
                if isinstance(self.window_generator, windowing.AdaptiveWindowGenerator):
                    self.window_generator.update_window_size(features_window_a, features_window_b)
                
                # compare both windows
                # result is true if a significant change is found
                result = self.population_comparer.compare(features_window_a, features_window_b)
                
                change_dictionary[window_b.end] = result

        change_series = pd.Series(change_dictionary)
        
        return change_series    

    def _get_change_points(self, series):
        """Gets a list of changepoints from a series of observations based on a threshold. 
        
        A change point is registered if the series values are below the threshold for the duration of 'min_observations_below'. 
        Change points will only be flagged if there is at least a distance of 'min_distance_change_streaks' between streaks of observations below the change point.
        
        Args:
            series: Pandas series with observations.
            
        Returns:
            List of change points.
        """
        # for each row, get whether its value is of threshold or lower
        series_below_threshold = series <= self.threshold

        # do an accumulative count of how many values below the threshold have been observed
        # restarts at 0 as soon as one value > threshold is observed
        below_threshold_counts = series_below_threshold * (series_below_threshold.groupby((series_below_threshold != series_below_threshold.shift()).cumsum()).cumcount() + 1)
        
        # reset the index of below_threshold_counts to 0...n instead of trace counts
        true_indices_series = below_threshold_counts.index
        below_threshold_counts = below_threshold_counts.reset_index(drop=True)

        # store change points as indices
        change_points = []
        
        # ix when the last streak ended
        last_change_streak_ended = None

        for index, streak_count in below_threshold_counts.iteritems():
            if streak_count < self.min_observations_below:
                continue
            
            # check if the streak is exactly the minimum number of observations
            if streak_count == self.min_observations_below:# or (streak_count > self.min_observations_below and streak_count % (2 * self.window_generator.window_size) == 0):
                
                integer_index_candidate = int(index - self.min_observations_below + 1)
                change_point_candidate = true_indices_series[integer_index_candidate]
                
                # definitely enter the change point if this is the first streak that was seen
                if last_change_streak_ended is None:
                    change_points.append(change_point_candidate)
                else:
                    # get distance to last streak end
                    distance_to_last_streak = integer_index_candidate - last_change_streak_ended - 1

                    if distance_to_last_streak >= self.min_distance_change_streaks:
                        change_points.append(change_point_candidate)
                            
            # update the end of the last change streak, if the current streak count exceeds the min observations below threshold
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

def get_all_attribute_drift_detectors(log, window_generator, population_comparer, level='trace', threshold=0.05, exclude_attributes=[], min_observations_below=3, min_distance_change_streaks=3):
    """Factory function to get attribute drift detectors for all trace level attributes in an event log.
        
    Args:
        log: A pm4py event log.
        window_generator: A windowing.WindowGenerator() to know which windowing strategy to use.
        pupulation_comparer: A pop_comparison.PopComparer() to know how to compare the populations.
        level: 'trace', 'event' or 'trace_and_event'.
        threshold: The threshold for change detection.
        exclude_attributes: Event log attributes for which no drift detector should be generated.
    
    Returns:
        List of drift detectors, one for each attribute.
    """
    
    # get all trace attributes
    trace_attributes = feature_extraction.get_all_trace_attributes(log)
    
    # remove attributes that shall be excluded
    trace_attributes.difference_update(exclude_attributes)

    # get all event attributes
    event_attributes = feature_extraction.get_all_event_attributes(log)
    
    # remove attributes that shall be excluded
    event_attributes.difference_update(exclude_attributes)
    
    # create the new feature extractors and detectors
    drift_detectors = []

    if level == 'trace' or level == 'trace_and_event':
        for attribute_name in trace_attributes:
            # get the unique feature extractor
            new_feature_extractor = feature_extraction.AttributeFeatureExtractor(attribute_level='trace', attribute_name=attribute_name)
            
            # create the drift detector
            drift_detector = DriftDetector(new_feature_extractor, window_generator, population_comparer, threshold=threshold, min_observations_below=min_observations_below, min_distance_change_streaks=min_observations_below)
            drift_detectors.append(drift_detector)
    if level == 'event' or level == 'trace_and_event':
        for attribute_name in event_attributes:
            # get the unique feature extractor
            new_feature_extractor = feature_extraction.AttributeFeatureExtractor(attribute_level='event', attribute_name=attribute_name)
            
            # create the drift detector
            drift_detector = DriftDetector(new_feature_extractor, window_generator, population_comparer, threshold=threshold, min_observations_below=min_observations_below, min_distance_change_streaks=min_observations_below)
            drift_detectors.append(drift_detector)
    
    return drift_detectors

