"""Module with functions to create windows of pm4py eventlogs.
"""

from abc import ABC, abstractmethod
import datetime
import numpy as np
import pm4py

from pm4py.objects.log.obj import EventLog
from pm4py.algo.filtering.log.timestamp import timestamp_filter

class Window():
    """Class to give context to a log window.
    
    Provides with the window's event log, start of window and end of window.
    """
    def __init__(self, log, start, end):
        """Create a new window.
        
        Args:
            log: Event log contained in this window.
            start: Start of this window.
            end: End of this window.
        """
        self.log = log
        self.start = start
        self.end = end

class WindowGenerator(ABC):
    """A window generator applies a windowing strategy such as fixed sized windows to create windows of an event log.
    """
    
    @abstractmethod
    def get_windows(self, event_log, start=None, end=None):
        """Create windows of the given log according to the initialization parameters.
        
        Args:
            event_log: An event log to create the windows for.
            start: Optional argument. If the start of the windowing should not be the first event in the log.
            end: Optional argument. To be set if the end of the windowing should not be the last complete window.
        
        Returns:
            Yields list of windows [(window_a, window_b), ...].
        """
        pass

class FixedWindowGenerator(WindowGenerator):
    """A fixes size window generator generates windows of fixed size.
    """
    
    def __init__(self,  window_size, window_offset=None, slide_by=1, unit_of_measure='trace', inclusion_criteria='events'):
        """Initialize the fixed sized window generator with the desired settings.
        
        Args:
            window_size: Size of each window as Python datetime.timedelta or trace number.
            window_offset: Offset of windows defined as datetime.timedelta or trace number. If None, offsets by window_size (non-overlapping windows).
            slide_by: How much to slide between generated windows. Defaults to 1.
            unit_of_measure: 'trace' or 'time'. Whether the window is created based on time or count of traces.
            inclusion_criteria: 'events', 'traces_intersecting', 'trace_contained'. Either return all events that take place in a window, all complete traces that have any event in the window or all complete traces that are fully contained in the window. Ignored if type is 'traces'.
        """
        # window size, type and inclusion criteria can be directly set
        self.window_size = window_size
        self.unit_of_measure = unit_of_measure
        self.inclusion_criteria = inclusion_criteria
        
        # set offset to window_size if it is not defined.
        if window_offset == None:
            window_offset = window_size
        self.window_offset = window_offset
        
        # set slide_by to 1 if it is not defined
        if slide_by == None:
            slide_by = 1
        self.slide_by = slide_by
        
    def get_windows(self, event_log, start=None, end=None):
        """Create windows of the given log according to the initialization parameters.
        
        Args:
            event_log: An event log to create the windows for.
            start: Optional argument. If the start of the windowing should not be the first event in the log.
            end: Optional argument. To be set if the end of the windowing should not be the last complete window.
        
        Returns:
            Yields list of windows [(window_a, window_b), ...].
        """
        
        # get start and end if they are not set
        if start == None:
            if self.unit_of_measure == 'trace': start = 0
            elif self.unit_of_measure == 'time': start = event_log[0][0]['time:timestamp'].replace(tzinfo=None)
        if end == None:
            if self.unit_of_measure == 'trace': end = len(event_log) - 1 # index of last trace is the end
            elif self.unit_of_measure == 'time': end = event_log[-1][-1]['time:timestamp'].replace(tzinfo=None)
        
        window_a_start = start
        window_a_end = window_a_start + self.window_size - 1 # TODO check if this -1 is wrong
        window_b_start = window_a_start + self.window_offset
        window_b_end = window_b_start + self.window_size - 1
        
        windows = {}

        while(window_b_end <= end):
            window_a_log = None
            window_b_log = None

            if self.unit_of_measure == 'time':
                if self.inclusion_criteria == 'events':
                    window_a_log = timestamp_filter.apply_events(event_log, window_a_start, window_a_end)
                    window_b_log = timestamp_filter.apply_events(event_log, window_b_start, window_b_end)
                elif self.inclusion_criteria == 'traces_contained':
                    window_a_log = timestamp_filter.filter_traces_contained(event_log, window_a_start, window_a_end)
                    window_b_log = timestamp_filter.filter_traces_contained(event_log, window_b_start, window_b_end)
                elif self.inclusion_criteria == 'traces_intersecting':
                    window_a_log = timestamp_filter.filter_traces_intersecting(event_log, window_a_start, window_a_end)
                    window_b_log = timestamp_filter.filter_traces_intersecting(event_log, window_b_start, window_b_end)
            elif self.unit_of_measure == 'trace':
                trace_window_a = event_log[window_a_start:window_a_end+1]
                trace_window_b = event_log[window_b_start:window_b_end+1]

                window_a_log = EventLog(trace_window_a, attributes=event_log.attributes, extensions=event_log.extensions, classifiers=event_log.classifiers,
                        omni_present=event_log.omni_present, properties=event_log.properties)
                window_b_log = EventLog(trace_window_b, attributes=event_log.attributes, extensions=event_log.extensions, classifiers=event_log.classifiers,
                        omni_present=event_log.omni_present, properties=event_log.properties)

            # package the windows for returning them
            window_a = Window(window_a_log, 
                              window_a_start,
                              window_a_end)
            window_b = Window(window_b_log, 
                              window_b_start,
                              window_b_end)

            windows = (window_a, window_b)
            
            # yield the result
            yield windows

            # set the new window start and end
            # doing this after the yield allows for any handling object to change the window size dynamically
            window_a_start = window_a_start + self.slide_by
            window_a_end = window_a_start + self.window_size - 1
            window_b_start = window_a_start + self.window_offset
            window_b_end = window_b_start + self.window_size - 1

class AdaptiveWindowGenerator(FixedWindowGenerator):
    """Adaptive window generator that sets the next window size according to the observed variability in an input variable.

    The approach is adapted from Maaradji et al. 2017.
    """
    
    def __init__(self,
        initial_window_size, 
        window_offset=None, 
        slide_by=1, 
        unit_of_measure='trace', 
        inclusion_criteria='events',
        min_window_size=2):
        """Initialize the variable sized window generator with the desired settings.
        
        Args:
            initial_window_size: Initial window size. Will increase/decrease adaptively. Python datetime.timedelta or trace number.
            window_offset: Offset of windows defined as datetime.timedelta or trace number. If None, offsets by window_size (non-overlapping windows).
            slide_by: How much to slide between generated windows. Defaults to 1.
            unit_of_measure: 'trace' or 'time'. Whether the window is created based on time or count of traces.
            inclusion_criteria: 'events', 'traces_intersecting', 'trace_contained'. Either return all events that take place in a window, all complete traces that have any event in the window or all complete traces that are fully contained in the window. Ignored if type is 'traces'.
            min_window_size: Minimum size of the adaptively set windows. Window sizes of smaller than 2 will not increase again.
        """
        super().__init__(initial_window_size, window_offset, slide_by, unit_of_measure, inclusion_criteria)

        self.initial_window_size = initial_window_size
        self._previous_variability = None
        self.min_window_size = min_window_size

    def _get_variability(self, features_window_a, features_window_b):
        """Get the variability as the number of observed unique values in the windows.
        
        Args:
            features_window_a: Extracted features from window a. numpy array.
            features_window_b: Extracted features from window b. numpy array.
        """
        features = np.concatenate([features_window_a, features_window_b])
        unique_values_in_windows = np.unique(features)
        number_unique_values_in_windows = len(unique_values_in_windows)

        # set the variability to at least 1, even if it was 0 to avoid window sizes of 0
        variability = max(1, number_unique_values_in_windows)

        return variability

    def update_window_size(self, features_window_a, features_window_b):
        """Update the window size by evaluating how the variability changed between windows.

        Args:
            features_window_a: Observed features in window a.
            features_window_b: Observed features in window b. 
        """
        # if there was no previous observation of variability, return
        if self._previous_variability is None:
            self._previous_variability = self._get_variability(features_window_a, features_window_b)
            return

        new_variability = self._get_variability(features_window_a, features_window_b)

        # the evolution ratio is the change in variability between two windows
        evolution_ratio = new_variability / self._previous_variability

        next_window_size = round(self.window_size * evolution_ratio)

        # set the next window size to min_window size at minimum
        next_window_size = max(self.min_window_size, next_window_size)

        # set the object variables
        self._previous_variability = new_variability
        self.window_size = next_window_size

    def reset_window_size(self):
        """Reset the window size to the initial setting."""
        self.window_size = self.initial_window_size
