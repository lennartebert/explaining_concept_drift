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
        
    def get_primary_and_secondary_change_series(self, event_log):
        """Get the changes registered in the primary and secondary drift detector as a Pandas Series.
        
        Args:
            event_log: A pm4py event log.
            
        Returns:
            (primary_change_series, secondary_change_series_dict): The primary change series and a dictionary with the change series for each secondary drift detector.
        """
        primary_change_series = self.primary_drift_detector.get_change_series(event_log)
        secondary_change_series_dict = {}
        for secondary_drift_detector in self.secondary_drift_detectors:
            secondary_change_points = secondary_drift_detector.get_change_series(event_log)
            secondary_change_series_dict[secondary_drift_detector.name] = secondary_change_points
        
        return primary_change_series, secondary_change_series_dict
    
    def attribute_importance_per_primary_change_point(self, event_log, max_distance=None):
        """Get the change points in the primary drift detector and all secondary change points that presume.
        
        Args:
            event_log: A pm4py event log.
            max_distance: The maximum distance between a primary and secondary changepoint so that the secondary change_point is returned.
            
        Returns:
            (primary_change_points, change_point_explanations): The primary change points and explanations for each.
        """
        # get process concept drift points
        primary_change_points = self.primary_drift_detector.get_change_points(event_log)
        
        # get secondary drifts
        # the resulting dictionary will have the format
        # {detector_name: change_point_list}, e.g., {'attribute XXX': [492, 2849]}
        
        all_secondary_change_points = {}
        for secondary_drift_detector in self.secondary_drift_detectors:
            secondary_change_points = secondary_drift_detector.get_change_points(event_log)
            all_secondary_change_points[secondary_drift_detector.name] = secondary_change_points
        
        # rank attributes by closest secondary change point before primary change point
        # for that, create a list of tuples with all secondary change points and detector names
        secondary_time_detector_tuples = []
        for drift_detector, change_points in all_secondary_change_points.items():
            for change_point in change_points:
                secondary_time_detector_tuples.append((change_point, drift_detector))
        
        # sort the secondary time detector tuples
        secondary_time_detector_tuples = sorted(secondary_time_detector_tuples)
        
        # get rank the secondary change point detectors by how close the secondary change point was to the primary change point
        change_point_explanations = {}
        for primary_change_point in primary_change_points:            
            distances_to_change_point = []
            for secondary_change_point, drift_detector in secondary_time_detector_tuples:
                distance = secondary_change_point - primary_change_point
                
                # keep distance in range max_distance
                if max_distance is not None and abs(distance) >= max_distance:
                    continue
                
                distances_to_change_point.append({
                    'detector': drift_detector,
                    'detector_change_point': secondary_change_point,
                    'distance': distance
                })
            
            # sort by change point distance
            distances_to_change_point = sorted(distances_to_change_point, key=lambda change_point_explanation: abs(change_point_explanation['distance']))
            
            change_point_explanations[primary_change_point] = distances_to_change_point
        
        return change_point_explanations


def plot_primary_and_secondary_change_series(primary_and_secondary_change_series, columns=2):
    """Plots the primary and secondary change series returned by DriftExplainer.get_primary_and_secondary_change_series(event_log).
    
    Args:
        primary_and_secondary_change_series: The primary and secondary change series as returned by DriftExplainer.get_primary_and_secondary_change_series(event_log).
    """
    # get the primary and secondary change
    primary_change_series = primary_and_secondary_change_series[0]
    secondary_change_series_dict = primary_and_secondary_change_series[1]
    
    n = len(secondary_change_series_dict.keys())
    rows = int(math.ceil(n / columns))

    gs = gridspec.GridSpec(rows, columns)
    fig = plt.figure(dpi=200, figsize = (8,8))
    
    # get sorted attribute list
    attribute_list = sorted(list(secondary_change_series_dict.keys()))
    
    # plot all secondary values
    for i, key in enumerate(attribute_list):
        series = secondary_change_series_dict[key]
        ax = fig.add_subplot(gs[i])
        ax.plot(series)
        ax.plot(primary_change_series, color='red', linestyle='dashed')
        # set the y axis to 0 - 1
        plt.ylim(0, 1)
        ax.title.set_text(key)
    
    # add a title
    fig.suptitle('Attribute Change over Time')

    fig.tight_layout()
    
    # create the legend
    legend_elements = [lines.Line2D([0], [0], color='red', linestyle='dashed', label='primary axis')]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05))
    
    plt.show()