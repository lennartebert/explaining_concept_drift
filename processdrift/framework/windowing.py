"""Module with functions to create windows of pm4py eventlogs.
"""

from abc import ABC, abstractmethod
import datetime

from pm4py.objects.log.obj import EventStream
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
    def get_windows(self, event_log):
        """Get windows from an event log. The window tuples are yielded for direct processing.
        
        Args:
            event_log: pm4py event log to be partitioned in windows
        """
        pass

class FixedSizeWindowGenerator(WindowGenerator):
    """A fixes size window generator generates windows of fixed size.
    """
    
    def __init__(self,  window_size, window_offset=None, slide_by=None, window_type='trace', inclusion_criteria='events'):
        """Initialize the fixed sized window generator with the desired settings.
        
        Args:
            window_size: Size of each window as Python datetime.timedelta or trace number.
            window_offset: Offset of windows defined as datetime.timedelta or trace number. If None, offsets by window_size (non-overlapping windows).
            slide_by: How much to slide between generated windows. Defaults to window_size.
            window_type: 
            window_type: 'trace' or 'time'. Whether the window is created based on time or count of traces.
            inclusion_criteria: 'events', 'traces_intersecting', 'trace_contained'. Either return all events that take place in a window, all complete traces that have any event in the window or all complete traces that are fully contained in the window. Ignored if type is 'traces'.
        """
        # window size, type and inclusion criteria can be directly set
        self.window_size = window_size
        self.window_type = window_type
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
            if self.window_type == 'trace': start = 0
            elif self.window_type == 'time': start = event_log[0][0]['time:timestamp'].replace(tzinfo=None)
        if end == None:
            if self.window_type == 'trace': end = len(event_log) - 1 # index of last trace is the end
            elif self.window_type == 'time': end = event_log[-1][-1]['time:timestamp'].replace(tzinfo=None)
        
        window_a_start = start
        window_a_end = window_a_start + self.window_size - 1
        window_b_start = window_a_start + self.window_offset
        window_b_end = window_b_start + self.window_size - 1
        
        windows = {}

        while(window_b_end <= end):
            window_a_log = None
            window_b_log = None

            if self.window_type == 'time':
                if inclusion_criteria == 'events':
                    window_a_log = timestamp_filter.apply_events(event_log, window_a_start, window_a_end)
                    window_b_log = timestamp_filter.apply_events(event_log, window_b_start, window_b_end)
                elif inclusion_criteria == 'traces_contained':
                    window_a_log = timestamp_filter.filter_traces_contained(event_log, window_a_start, window_a_end)
                    window_b_log = timestamp_filter.filter_traces_contained(event_log, window_b_start, window_b_end)
                elif inclusion_criteria == 'traces_intersecting':
                    window_a_log = timestamp_filter.filter_traces_intersecting(event_log, window_a_start, window_a_end)
                    window_b_log = timestamp_filter.filter_traces_intersecting(event_log, window_b_start, window_b_end)
            elif self.window_type == 'trace':
                # ignore the inclusion criteria (only makes sense for the time-based approach)
                window_a_log = event_log[window_a_start:window_a_end+1]
                window_b_log = event_log[window_b_start:window_b_end+1]

            # set the new window start and end
            window_a_start = window_a_start + self.slide_by
            window_a_end = window_a_start + self.window_size - 1
            window_b_start = window_a_start + self.window_offset
            window_b_end = window_b_start + self.window_size -1

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

# TODO implement AdaptiveWindowGenerator