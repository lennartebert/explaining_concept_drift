"""Module for the detection of drift. Can be used for either the primary or secondary change perspective.
"""

import subprocess
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pm4py
from matplotlib import lines
from processdrift.explanation import (feature_extraction,
                                      population_comparison, windowing)


class DriftDetector(ABC):
    """The DriftDetector determines the change series and change points for a given event log."""
    @abstractmethod
    def get_changes(self, event_log, around_change_points=None, max_distance=None, unit_of_measure='trace'):
        """Get the DriftDetectionResult for an event log. All further detail is specified in the inheriting classes.

        Args:
            event_log: A pm4py event log.
            around_change_points: Only get changes at a maximum of max_distance around an array of change_points.
            max_distance: Maximum distance of change to any change point.
            unit_of_measure: 'trace' or 'time'; Whether the change point and maximum distance are defined as timestamps or trace counts.

        Returns:
            A DriftDetectionResult object with a change series and the change points.
        """

    @property
    @abstractmethod
    def name(self):
        pass


class HypothesisTestDD(DriftDetector):
    """The drift detector gets a feature's change over time and can retrieve the according change points.
    """

    def __init__(self, feature_extractor, window_generator, population_comparer, change_point_extractor, detector_name=None):
        """Create a new drift detector and supply strategies for feature extraction, window generation and population comparison.

        Args:
            feature_extractor: A feature extractor from processdrift.explanation.feature_extraction.
            window_generator: A window generator from processdrift.explanation.windowing.
            population_comparer: A population comparer from processdrift.explanation.popcomparison.
            change_point_extractor: The change point extractor used to extract change points from the change series.
            detector_name: The name of the detector.
        """
        self.feature_extractor = feature_extractor
        self.window_generator = window_generator
        self.population_comparer = population_comparer
        self.change_point_extractor = change_point_extractor
        self._name = detector_name

    @property
    def name(self):
        # The detector name is taken from the feature extractor if not otherwise specified
        if self._name == None:
            return self.feature_extractor.name
        else:
            return self._name

    def get_changes(self, event_log, around_change_points=None, max_distance=None, unit_of_measure='trace'):
        """Get the DriftDetectionResult for an event log through a hypothesis test based approach.

        The search for changes can be restricted to the area of max_distance around traces specified in a list.

        Args:
            event_log: A pm4py event log.
            around_change_points: Only get changes at a maximum of max_distance around an array of change_points.
            max_distance: Maximum distance of change to any change point.
            unit_of_measure: 'trace' or 'time'; Whether the change point and maximum distance are defined as timestamps or trace counts.

        Returns:
            A DriftDetectionResult object with a change series and the change points.
        """
        if unit_of_measure == 'time':
            # TODO implement time as unit of measure
            raise NotImplementedError(
                f'Unit of measure {unit_of_measure} not implemented.')

        # get the change series
        change_series = self._get_change_series(
            event_log, around_change_points=around_change_points, max_distance=max_distance)

        # get the change points
        change_points = self.change_point_extractor.get_change_points(
            change_series)

        result = DriftDetectionResult(
            name=self.name, change_points=change_points, change_series=change_series)

        return result

    def _get_change_series(self, event_log, around_change_points, max_distance, unit_of_measure='trace'):
        """Get the change over time from the event log.

        The search for changes can be restricted to the area of max_distance around traces specified in a list.

        Args:
            event_log: A pm4py event log.
            around_change_points: Only get changes at a maximum of max_distance around an array of change_points.
            max_distance: Maximum distance of change to any change point.
            unit_of_measure: 'trace' or 'time'; Whether the change point and maximum distance are defined as timestamps or trace counts.

        Returns:
            The comparison result over time.
        """
        if unit_of_measure == 'time':
            # TODO implement time as unit of measure
            raise NotImplementedError(
                f'Unit of measure {unit_of_measure} not implemented.')

        change_dictionary = {}

        # if the user specified change points, only search around these
        if around_change_points is not None:
            # make sure that around_change_points is sorted
            around_change_points.sort()

            last_end_change_point_window = None
            for change_point in around_change_points:
                # determine the area in which to look for changes
                start_change_point_window = max(0, change_point-max_distance)
                end_change_point_window = min(
                    len(event_log), change_point+max_distance)

                # reset the size of the adaptive window generator if there was a gap between the last window around the change point and this one
                if last_end_change_point_window is not None:
                    # did the windows overlap -> set start_change_point_window to end of last change point window
                    if last_end_change_point_window > start_change_point_window:
                        start_change_point_window = last_end_change_point_window
                    # else, keep it as is and reset the adaptive window generator window size, if that window generator is used
                    else:
                        # update window size for adaptive generator
                        if isinstance(self.window_generator, windowing.AdaptiveWG):
                            self.window_generator.reset_window_size()
                # update the last end of the change point window
                last_end_change_point_window = end_change_point_window

                window_generator_start = max(
                    0, start_change_point_window-2*self.window_generator.window_size+1)

                # get windows for comparison
                for window_a, window_b in self.window_generator.get_windows(event_log, start=window_generator_start):
                    if window_b.end > end_change_point_window:
                        break

                    # get features for each window
                    features_window_a = self.feature_extractor.extract(
                        window_a.event_log)
                    features_window_b = self.feature_extractor.extract(
                        window_b.event_log)

                    # update window size for adaptive generator
                    if isinstance(self.window_generator, windowing.AdaptiveWG):
                        self.window_generator.update_window_size(
                            features_window_a, features_window_b)

                    # compare both windows
                    # result is true if a significant change is found
                    result = self.population_comparer.compare(
                        features_window_a, features_window_b)

                    change_dictionary[window_b.end] = result
        # look for change globally, not just around change points
        else:
            # get windows for comparison
            for window_a, window_b in self.window_generator.get_windows(event_log):
                # get features for each window
                features_window_a = self.feature_extractor.extract(
                    window_a.event_log)
                features_window_b = self.feature_extractor.extract(
                    window_b.event_log)

                # update window size for adaptive generator
                if isinstance(self.window_generator, windowing.AdaptiveWG):
                    self.window_generator.update_window_size(
                        features_window_a, features_window_b)

                # compare both windows
                # result is true if a significant change is found
                result = self.population_comparer.compare(
                    features_window_a, features_window_b)

                change_dictionary[window_b.end] = result

        change_series = pd.Series(change_dictionary)

        return change_series


class ProDriftDD(DriftDetector):
    """Uses Apromore ProDrift 2.5 for drift detection.

    ProDrift 2.5 can be downloaded from: https://apromore.com/research-lab/.
    """

    def __init__(self,
                 path_to_prodrift,
                 drift_detection_mechanism='runs',
                 window_size=200,
                 window_mode='adaptive',
                 detect_gradual_as_well=False):
        """Create a new ProDrift Drift detector.

        Args:
            path_to_prodrift: File path to the ProDrift .jar file.
            drift_detection_mechanism: Which ProDrift detection mechanism to use. Uses the runs feature drift detection by default.
            window_size=200: Size of each window.
            window_mode='adaptive': Whether fixed or adaptive windows are used.
            detect_gradual_as_well: Should gradual drift be detected as well?        
        """
        self.path_to_prodrift = path_to_prodrift
        self.drift_detection_mechanism = drift_detection_mechanism
        self.window_size = window_size
        self.window_mode = window_mode
        self.detect_gradual_as_well = detect_gradual_as_well

    def get_changes(self, event_log, around_change_points=None, max_distance=None, unit_of_measure=None):
        """Get the DriftDetectionResult for an event log by running ProDrift.

        Args:
            event_log: A pm4py event log.
            around_change_points: Not used.
            max_distance: Not used.
            unit_of_measure: 'trace' or 'time'; Not used.

        Returns:
            A DriftDetectionResult object with the change points.
        """

        # get the change points
        change_points = self._get_change_points(event_log)

        result = DriftDetectionResult(
            name=self.name, change_points=change_points, change_series=None)

        return result

    def _get_change_points(self, event_log):
        """Use ProDrift to get the change points from an event log.

        Args:
            event_log: A PM4Py event log.

        Returns:
            List of change points.
        """
        # save event log to temporary file
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            # save the event log in a temporary directory
            event_log_path = os.path.join(tmp, 'event_log.xes')
            pm4py.objects.log.exporter.xes.exporter.apply(
                event_log, event_log_path)

            # build the ProDrift command
            command = f"java -jar \"{self.path_to_prodrift}\" -fp \"{event_log_path}\"" \
                f" -ddm {self.drift_detection_mechanism}"

            if self.window_size is not None:
                command += f" -ws {self.window_size}"

            if self.window_mode is not None:
                if self.window_mode == 'adaptive':
                    pass  # adaptive mode is the default
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
                    line_starting_with_trace_number = line.split(
                        'drift detected at trace: ')[1]
                    trace_number = line_starting_with_trace_number.split(' ')[
                        0]
                    change_points.append(int(trace_number))

            return change_points

    @property
    def name(self):
        return 'ProDrift Drift Detector'


class TrueKnownDD(DriftDetector):
    """A drift detector to be used if the true change points are known.

    Will always return 100% accurate results.
    """

    def __init__(self, change_points):
        """Create a new drift detector and supply strategies for feature extraction, window generation and population comparison.

        Args:
            change_points: List of change points.
        """
        self.change_points = change_points

    def get_changes(self, event_log, around_change_points=None, max_distance=None, unit_of_measure=None):
        """Get the DriftDetectionResult from a pre-specified change point list.

        Args:
            event_log: A pm4py event log.
            around_change_points: Not used.
            max_distance: Not used.
            unit_of_measure: 'trace' or 'time'; Not used.

        Returns:
            A DriftDetectionResult object with a the change points.
        """

        result = DriftDetectionResult(
            name=self.name, change_points=self.change_points, change_series=None)
        return result

    def _get_change_points(self, event_log):
        """Get the change points as presented when initialized.

        Args:
            event_log: A pm4py event log.

        Returns:
            List of change points.
        """
        return self.change_points

    @property
    def name(self):
        return 'True Known Drift Detector'


class DriftDetectionResult():
    """Results object for the change point detector."""

    def __init__(self, name=None, change_points=None, change_series=None):
        """The result of running a drift detector.

        Args:
            name: Name of the drift detector.
            change_points: List of change points.
            change_series: Result series from window comparison.
        """
        self.name = name
        self.change_points = change_points
        self.change_series = change_series

    def plot(self, primary_change_points=None, threshold=None, start=None, end=None, offset_legend=-0.87, ylabel='p-value'):
        """
        Plots the drift detection result

        Args:
            primary_change_points: Optional; Plot the primary change points. Specify as list of trace counts.
            threshold: Optional; Show a threshold line. Specify the threshold value.
            start: Trace number where to start the plot.
            end: Trace number where to end the plot.
            offset_legend: Vertical offset of the plot's legend.
            ylabel: Label of y-axis.

        Returns:
            Plot of drift detection result.
        """
        plt.figure(dpi=300, figsize=(4, 2))

        # get start and end of change series if desired
        selected_change_series = self.change_series
        if start is not None and end is not None:
            selected_change_series = selected_change_series.loc[start:end]
        elif start is not None and end is None:
            selected_change_series = selected_change_series.loc[start:]
        elif start is None and end is not None:
            selected_change_series = selected_change_series.loc[:end]

        plt.plot(selected_change_series, color='blue',
                 marker=".", linewidth=0.75, markersize=1,
                 label=f"p-value")

        # plot the primary change points if some where defined
        if primary_change_points is not None:
            # plot the change points by a red line
            for pcp in primary_change_points:
                plt.axvline(x=pcp, color='red', linewidth=1)

        # plot the secondary change points if some where defined
        if self.change_points is not None:
            # plot the secondary change points with crosses
            x = self.change_points
            # check if there were secondary change points
            if x is not None:
                y = list(selected_change_series.loc[self.change_points])
                plt.scatter(x=x,
                            y=y,
                            marker='x',
                            color='black'
                            )

        # plot the threshold as a grey line
        if threshold is not None:
            plt.axhline(y=threshold, color='grey', linewidth=1)

        plt.ylabel(ylabel)
        plt.xlabel('traces')
        plt.title(f'{self.name}')

        plt.ylim(-0.01, 1.1)
        plt.xlim(min(selected_change_series.index) - 10,
                 max(selected_change_series.index) + 10)

        # create the legend
        legend_elements = []
        if primary_change_points is not None:
            legend_elements.append(lines.Line2D(
                [0], [0], color='red', linestyle='solid', label='Primary Change Points'))

        legend_elements.append(lines.Line2D(
            [0], [0], color='blue', marker=".", linewidth=0.75, markersize=1, label='Secondary Change Series'))

        if threshold is not None:
            legend_elements.append(lines.Line2D(
                [0], [0], color='grey', linewidth=1, label=f'Threshold'))

        if self.change_points is not None:
            legend_elements.append(lines.Line2D(
                [0], [0], color='black', marker='x', linestyle='None', label=f'Detected Secondary Change Points'))

        plt.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, offset_legend))

    def __repr__(self):
        result = f'Results for Drift Detector {self.name}:'
        result = f'{result}\nChange points: {self.change_points}'
        if self.change_series is not None:
            result = f'{result}\nHas a change series.'
        else:
            result = f'{result}\nHas no change series.'
        return result


def get_attribute_drift_detectors(attribute_level_types, window_generator, change_point_extractor, min_samples_per_test=5):
    """Factory function to get attribute drift detectors for all trace level attributes in an event log.

    Args:
        attribute_level_types:  List of triplets of attribute name, attribute level and type.
        window_generator: A windowing.WindowGenerator() to know which windowing strategy to use.
        change_point_extractor: A change_detection.ChangePointExtractor() to get change points from the change series.
        min_samples_per_test: Minimum number of samples for each observed attribute value for hypothesis tests.

    Returns:
        List of drift detectors, one for each attribute.
    """

    # create the new feature extractors and detectors
    drift_detectors = []

    for attribute_name, attribute_level, attribute_type in attribute_level_types:
        # create the feature extractor
        feature_extractor = feature_extraction.AttributeFE(
            attribute_level=attribute_level, attribute_name=attribute_name)

        # create the population comparer
        population_comparer = None
        if attribute_type == 'categorical':
            population_comparer = population_comparison.GTestPC(
                min_samples_per_test)
        else:
            population_comparer = population_comparison.KSTestPC()

        # create the drift detector
        drift_detector = HypothesisTestDD(
            feature_extractor, window_generator, population_comparer, change_point_extractor)
        drift_detectors.append(drift_detector)

    return drift_detectors
