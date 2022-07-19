"""Drift explainer module to explain process mining concept drift based on attribute value drift.
"""

import math
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import lines


class DriftExplainer():
    """The process mining concept drift explainer identifies drift in event logs and returns the secondary features which also showed drift behavior. The drift explainer is used as assistence for manual root cause analysis.
    """
    
    def __init__(self, primary_drift_detector, secondary_drift_detectors):
        """Create a new DriftExplainer given a primary drift detector and a list of secondary drift detectors.
        
        Args:
            primary_drift_detector: The primary DriftDetector. E.g., a drift detector that detects drift on control flow level.
            secondary_drift_detectors: The list of secondary DriftDetectors. E.g., drift detectors that identify drift in trace level process attributes.
        """
        self.primary_drift_detector = primary_drift_detector
        self.secondary_drift_detectors = secondary_drift_detectors

    def get_possible_drift_explanations(self, event_log, max_distance=None):
        """Gets a drift explainer results object with the possible change explanations and observed changes in the primary and secondary detectors.
        
        Args:
            event_log: A pm4py event log.
            max_distance: Maximum distance between a primary and secondary change point so that the secondary change point is evaluated.
            
        Returns:
            DriftExplanationResult with possible drift explanations and the results of the primary and secondary perspective drift detectors.
        """
        primary_dd_result = self.primary_drift_detector.get_changes(event_log)
        primary_change_points = primary_dd_result.change_points
    
        # for each secondary drift detector, get changes
        secondary_dd_result_dictionary = {}
        for secondary_drift_detector in self.secondary_drift_detectors:
            secondary_dd_result = None

            # pass the primary changes, depending on whether the max_distance was set
            if max_distance is not None:
                secondary_dd_result = secondary_drift_detector.get_changes(event_log, primary_change_points, max_distance)
            else:
                secondary_dd_result = secondary_drift_detector.get_changes(event_log)

            secondary_dd_result_dictionary[secondary_drift_detector.name] = secondary_dd_result
        
        # get the possible drift point explanations
        possible_drift_explanations = self._get_possible_drift_explanations(primary_change_points, secondary_dd_result_dictionary, max_distance)

        # package results into results object
        result = DriftExplanationResult(primary_dd_result, secondary_dd_result_dictionary, possible_drift_explanations)
        return result

    def _get_possible_drift_explanations(self, primary_change_points, secondary_dd_result_dictionary, max_distance):
        """
        TODO: Complete docstring
        
        Returns:
            (primary_change_points, change_point_explanations): The primary change points and explanations for each.
        """
        
        all_secondary_change_points = {}
        for detector_name, secondary_changes in secondary_dd_result_dictionary.items():
            secondary_change_points = secondary_changes.change_points
            all_secondary_change_points[detector_name] = secondary_change_points

        # rank attributes by closest secondary change point before primary change point
        # for that, create a list of tuples with all secondary change points and detector names
        secondary_time_detector_tuples = []
        for drift_detector_name, change_points in all_secondary_change_points.items():
            for change_point in change_points:
                secondary_time_detector_tuples.append((change_point, drift_detector_name))
        
        # sort the secondary time detector tuples
        secondary_time_detector_tuples = sorted(secondary_time_detector_tuples)
        
        # get rank the secondary change point detectors by how close the secondary change point was to the primary change point
        change_point_explanations = {}
        for primary_change_point in primary_change_points:            
            distances_to_change_point = []
            for secondary_change_point, drift_detector in secondary_time_detector_tuples:
                distance = secondary_change_point - primary_change_point
                
                # keep distance in range max_distance
                if max_distance is not None and abs(distance) > max_distance:
                    continue
                
                distances_to_change_point.append({
                    'detector': drift_detector,
                    'change_point': secondary_change_point,
                    'distance': distance
                })
            
            # sort by change point distance
            distances_to_change_point = sorted(distances_to_change_point, key=lambda change_point_explanation: abs(change_point_explanation['distance']))
            
            change_point_explanations[primary_change_point] = distances_to_change_point
        
        return change_point_explanations

class DriftExplanationResult():
    """Results object for the drift explanations."""
    def __init__(self, primary_dd_result, secondary_dd_result_dictionary, possible_drift_explanations):
        self.primary_dd_result = primary_dd_result
        self.secondary_dd_result_dictionary = secondary_dd_result_dictionary
        self.possible_drift_explanations = possible_drift_explanations
    
    def plot(self, columns=2,
        plot_primary_change_series=False,
        plot_annotations=False,
        threshold=0.05,
        offset_legend=-0.15):
        """Plots the primary and secondary change series returned by DriftExplainer.get_primary_and_secondary_changes(event_log).
    
        Args:
            columns: Number of columns
            plot_primary_change_series: Whether or not to plot the primary change series.
            plot_annotations: Whether or not to plot annotations for each change point.
            threshold: Value or p-value threshold.
            offset_legend: Vertical offset of the plot's legend.
        
        Returns:
            Plot.
        """
        # get the primary and secondary change
        primary_change_points = self.primary_dd_result.change_points
        primary_change_series = self.primary_dd_result.change_series
        secondary_change_dict = self.secondary_dd_result_dictionary
        
        n = len(secondary_change_dict.keys())
        rows = max(1, int(math.ceil(n / columns))) # set the row count to at least 1

        gs = gridspec.GridSpec(rows, columns)
        fig = plt.figure(dpi=200, figsize = (8,2*rows))
        
        # get sorted attribute list
        attribute_list = sorted(list(secondary_change_dict.keys()))
        
        # plot all secondary values
        for i, attribute_name in enumerate(attribute_list):
            secondary_change_series = secondary_change_dict[attribute_name].change_series
            secondary_change_points = secondary_change_dict[attribute_name].change_points

            ax = fig.add_subplot(gs[i])
            
            # plot the change points by a red line
            for change_point in primary_change_points:
                plt.axvline(x=change_point, color='red', linewidth=1)

            # plot the threshold as a grey line
            if threshold is not None:
                plt.axhline(y=threshold, color='grey', linewidth=1)

            # based on user choice, plot the primary change series
            if plot_primary_change_series:
                ax.plot(primary_change_series, color='red', linestyle='dashed')

            ax.plot(secondary_change_series)
            
            x = secondary_change_points
            # check if there were secondary change points
            if x is not None:
                y = list(secondary_change_series.loc[secondary_change_points])
                ax.scatter(x=x, 
                    y=y,
                    marker='x',
                    color='black'
                )
            
            if plot_annotations:
                for i, secondary_change_point in enumerate(x):
                    change_point_trace = x[i]
                    change_point_y = y[i]
                    ax.annotate(f'({change_point_trace}, {change_point_y:.2f})', (x[i], y[i]))

            # set the y axis so all changes in the data can be clearly seen
            plt.ylim(-0.1, 1.1)
            plt.xlim(0, primary_change_series.index[-1])
            ax.title.set_text(attribute_name)

            # give a y label to all plots in first column
            if i % columns == 0:
                ax.set_ylabel('p-value')

            # give x labels to last row
            # e.g. 2 columns and 3 rows
            # 1, 2, 3, 4 -> FALSE; 5, 6 -> True
            if math.ceil((i+1)/columns) == rows:
                ax.set_xlabel('traces')
        
        # add a title
        fig.suptitle('Attribute Change over Time')

        fig.tight_layout()
        
        # create the legend
        legend_elements = [lines.Line2D([0], [0], color='red', linestyle='solid', label='Primary Change Points'),
            lines.Line2D([0], [0], color='grey', linestyle='solid', label=f'Threshold ({threshold})'),
            lines.Line2D([0], [0], color='black', marker='x', linestyle='None', label=f'Detected Secondary Change Points')
        ]

        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, offset_legend))
        
        plt.show()

        return plt
