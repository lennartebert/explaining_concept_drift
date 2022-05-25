"""Implementation of the concept drift explanation framework.
"""
import pandas as pd

class ConceptDriftExplainer():
    """The concept drift explainer explains concept drift which is observable in an event log by ranking attributes by their importance.
    """
    
    def __init__(self, drift_point_detector, attribute_importance_measurer):
        self.drift_point_detector = drift_point_detector
        self.attribute_importance_measurer = attribute_importance_measurer
    
    def get_attribute_importance_per_changepoint(self, event_log):
        # get process concept drift points
        change_points = self.drift_point_detector.get_change_points(event_log)
        
        attribute_importance_per_changepoint = {}
        for change_point_location in change_points:
            attribute_importance_series = self.attribute_importance_measurer.get_attribute_importance(event_log, change_point_location)
            attribute_importance_per_changepoint[change_point_location] = attribute_importance_series
        
        return attribute_importance_per_changepoint
    
    def get_attribute_importance(self, event_log):
        # get the attribute importance per change point
        attribute_importance_per_changepoint = self.get_attribute_importance_per_changepoint(event_log)
        
        # merge the importance scores
        # TODO could also implement different strategies for this
        attribute_importance_df = pd.concat(list(attribute_importance_per_changepoint.values()), axis=1)
        summed_attribute_scores = attribute_importance_df.sum(axis=1)
        
        # sort the summed attribute scores
        summed_attribute_scores = summed_attribute_scores.sort_values(ascending=False)
        
        return summed_attribute_scores