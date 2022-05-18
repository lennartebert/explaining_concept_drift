from abc import ABC, abstractmethod

class DriftPointDetector(ABC):
    @abstractmethod
    def get_change_points(self, event_log):
        pass
    
class DriftPointDetectorTrueKnown(DriftPointDetector):
    """A drift point detector to work for synthetic data where the drift point is known.
    
    Used only for testing and evaluation purposes.
    """
    def __init__(self, true_drift_points):
        """Initialize the drift point detector with the known list of drift points.
        
        Args:
            true_drift_points: List of known process drift points. E.g., [100, 200, 500]
        """
        self.true_drift_points = true_drift_points
    
    def get_change_points(self, event_log=None):
        """Returns the list of known change points for the event log.
        
        Args:
            event_log: An event log. The event log is not considered for this drift detector and can also be None.
        
        Returns:
            The list of known drift points.
        """
        return self.true_drift_points