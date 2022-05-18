"""Module for window creators used in the process drift explanation framework.
"""

from concept_drift import windows

class WindowCreator():
    """Base implementation of a window creater with many parameters to set"""
    def __init__(self,  window_size, window_offset=None, slide_by=None, start=None, end=None, inclusion_criteria='events'):
        """Initialize the window creator with the desired settings.
        
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
        
    def create_windows(self, event_log, start=None, end=None):
        """Create windows of the given log according to the initialization parameters.
        
        Args:
            event_log: An event log to create the windows for.
            start: Optional argument. If the start of the windowing should not be the first event in the log.
            end: Optional argument. To be set if the end of the windowing should not be the last complete window.
        
        Returns:
            List of windows in the format [(window_a_start, (window_a, window_b)), (window_x_start, (window_x, window_y))...]
        """
        created_windows = windows.get_log_windows(event_log, self.window_size, self.window_offset, self.slide_by, start, end, self.inclusion_criteria)
        return created_windows