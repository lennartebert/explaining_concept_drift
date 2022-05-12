"""Module with functions to create windows of pm4py eventlogs.
"""

from pm4py.objects.log.obj import EventStream
import datetime
from pm4py.algo.filtering.log.timestamp import timestamp_filter


def get_log_windows(log, window_size, window_offset=None, slide_by=None, start=None, end=None, inclusion_criteria='events', type=None):
    """Cut a pm4py event log into windows.

    Args:
        log: A pm4py EventLog.
        window_size: Size of each window as Python datetime.timedelta or trace number.
        window_offset: Offset of windows defined as datetime.timedelta or trace number. If None, offsets by window_size (non-overlapping windows).
        slide_by: How much to slide between generated windows. Defaults to window_size.
        start: Optional argument. If the start of the windowing should not be the first event in the log.
        end: Optional argument. To be set if the end of the windowing should not be the last complete window.
        inclusion_criteria: 'events', 'traces_intersecting', 'trace_contained'. Either return all events that take place in a window, all complete traces that have any event in the window or all complete traces that are fully contained in the window. Ignored if type is 'traces'.
        type: 'traces' or 'time'. Whether the window is defined by trace indexes or timesteps. If None, will be inferred automatically.

    Returns:
        List of windows in the format [(window_a_start, (window_a, window_b)), (window_x_start, (window_x, window_y))...]
    """
    if type == None:
        if isinstance(window_size, datetime.timedelta): # TODO could implement additional checks for types of other date fields
            type = 'time'
        else:
            type = 'traces'

     # set offset to window_size if it is not defined.
    if window_offset == None:
        window_offset = window_size

    # set slide_by to window_size if it is not defined
    if slide_by == None:
        slide_by = window_size

    if start == None:
        if type == 'traces': start = 0
        elif type == 'time': start = log[0][0]['time:timestamp'].replace(tzinfo=None)
    if end == None:
        if type == 'traces': end = len(log) - 1 # index of last trace is the end
        elif type == 'time': end = log[-1][-1]['time:timestamp'].replace(tzinfo=None)

    window_a_start = start
    window_a_end = window_a_start + window_size
    window_b_start = window_a_start + window_offset
    window_b_end = window_b_start + window_size

    windows = {}

    while(window_b_end <= end):
        window_a = None
        window_b = None

        if type == 'time':
            if inclusion_criteria == 'events':
                window_a = timestamp_filter.apply_events(log, window_a_start, window_a_end)
                window_b = timestamp_filter.apply_events(log, window_b_start, window_b_end)
            elif inclusion_criteria == 'traces_contained':
                window_a = timestamp_filter.filter_traces_contained(log, window_a_start, window_a_end)
                window_b = timestamp_filter.filter_traces_contained(log, window_b_start, window_b_end)
            elif inclusion_criteria == 'traces_intersecting':
                window_a = timestamp_filter.filter_traces_intersecting(log, window_a_start, window_a_end)
                window_b = timestamp_filter.filter_traces_intersecting(log, window_b_start, window_b_end)
        elif type == 'traces':
            # ignore the inclusion criteria (only makes sense for the time-based approach)
            window_a = log[window_a_start:window_a_end+1]
            window_b = log[window_b_start:window_b_end+1]

        windows[window_a_start] = (window_a, window_b)

        window_a_start = window_a_start + slide_by
        window_a_end = window_a_start + window_size
        window_b_start = window_a_start + window_offset
        window_b_end = window_b_start + window_size
    
    return windows
