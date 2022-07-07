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

    def get_primary_and_secondary_changes(self, event_log, max_distance=None):
        """Get changes in the primary and secondary drift detectors.

        Args:
            event_log: A pm4py event log.
            max_distance: Maximum distance between a primary and secondary change point so that the secondary change point is evaluated.
            
        Returns:
            (primary_changes, secondary_changes): Dictionaries for primary and secondary changes.
        """
        primary_changes = self.primary_drift_detector.get_changes(event_log)
        primary_change_points = primary_changes['change_points']

        # for each secondary drift detector, get changes
        secondary_changes_dictionary = {}
        for secondary_drift_detector in self.secondary_drift_detectors:
            secondary_changes = None
            
            # pass the primary changes, depending on whether the max_distance was set
            if max_distance is not None:
                secondary_changes = secondary_drift_detector.get_changes(event_log, primary_change_points, max_distance)
            else:
                secondary_changes = secondary_drift_detector.get_changes(event_log)

            secondary_changes_dictionary[secondary_drift_detector.name] = secondary_changes
        
        return primary_changes, secondary_changes_dictionary


def get_possible_change_explanations(primary_and_secondary_changes, max_distance=None):
        """Get the change points in the primary drift detector and all secondary change points that presume.
        
        Args:
            event_log: A pm4py event log.
            max_distance: The maximum distance between a primary and secondary changepoint so that the secondary change_point is returned.
            
        Returns:
            (primary_change_points, change_point_explanations): The primary change points and explanations for each.
        """

        primary_change_points = primary_and_secondary_changes[0]['change_points']
        secondary_changes_dict = primary_and_secondary_changes[1]
        
        all_secondary_change_points = {}
        for detector_name, secondary_changes in secondary_changes_dict.items():
            secondary_change_points = secondary_changes['change_points']
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

def plot_primary_and_secondary_changes(primary_and_secondary_change,
    columns=2,
    plot_primary_change_series=False,
    plot_annotations=False,
    threshold=0.05):
    """Plots the primary and secondary change series returned by DriftExplainer.get_primary_and_secondary_changes(event_log).
    
    Args:
        primary_and_secondary_change: The primary and secondary change series as returned by DriftExplainer.get_primary_and_secondary_changes(event_log).
        columns: Number of columns
        plot_primary_change_series: Whether or not to plot the primary change series.
        plot_annotations: Whether or not to plot annotations for each change point.
        threshold: Value or p-value threshold.
    
    Returns:
        Plot.
    """
    # get the primary and secondary change
    primary_change_points = primary_and_secondary_change[0]['change_points']
    primary_change_series = primary_and_secondary_change[0]['change_series']
    secondary_change_dict = primary_and_secondary_change[1]
    
    n = len(secondary_change_dict.keys())
    rows = max(1, int(math.ceil(n / columns))) # set the row count to at least 1

    gs = gridspec.GridSpec(rows, columns)
    fig = plt.figure(dpi=200, figsize = (8,2*rows))
    
    # get sorted attribute list
    attribute_list = sorted(list(secondary_change_dict.keys()))
    
    # plot all secondary values
    for i, attribute_name in enumerate(attribute_list):
        secondary_change_series = secondary_change_dict[attribute_name]['change_series']
        secondary_change_points = secondary_change_dict[attribute_name]['change_points']

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

    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15))
    
    plt.show()

    return plt

def multiplot_primary_and_secondary_changes(primary_and_secondary_change, secondary_drict_detectors_per_plot, *args, **kwargs):
    """Create multiple plots for primary and secondary changes.

    This function is useful because many features can lead to plots that are too big for printing on one page.
    
    See plot_primary_and_secondary_changes() for documentation of *args and **kwargs.

    Args:
        primary_and_secondary_change: primary and secondary change results.
        secondary_drict_detectors_per_plot: How many secondary change results to put per plot.
    """
    primary_changes = primary_and_secondary_change[0]
    secondary_changes = primary_and_secondary_change[1]

    def slice_per(source, step):
        list_of_lists = [source[x:x+step] for x in range(0, len(source), step)]
        return list_of_lists
    
    secondary_change_packages = slice_per(list(secondary_changes.keys()), secondary_drict_detectors_per_plot)
    
    plots = []
    for secondary_change_package in secondary_change_packages:
        temp_primary_and_secondary_change = (primary_changes, {key: secondary_changes[key] for key in secondary_change_package})
        plot = plot_primary_and_secondary_changes(temp_primary_and_secondary_change, *args, **kwargs)
        plots.append(plot)
    
    return plots