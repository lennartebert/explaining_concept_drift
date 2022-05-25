"""Module with all classes and methods for feature extraction.
"""

from abc import ABC, abstractmethod
import numpy as np

from processdrift import features

class FeatureExtractor(ABC):
    """A feature extractor implements "how" features are extracted. By calling ´extract(log)´ with a log file, the feature extraction strategy is applied.
    """
    
    def __init__(self, name):
        """Initialize the feature extractor and set its name.
        
        Args:
            name: Name of the feature extractor.
        """
        self.name = name
    
    @abstractmethod
    def extract(self, log):
        """Apply a feature extraction strategy to extract features from a pm4py event log.
        
        Args:
            log: A pm4py event log to extract features from.
        
        Returns:
            Extracted features. TODO: Should this be limited to numpy Array/Matrix?
        """
        pass
    
    def __repr__(self):
        return self.name


class AttributeFeatureExtractor(FeatureExtractor):
    """The attribute feature extractor retrieves all attribute values from an event log.
    """
    def __init__(self, attribute_level, attribute_name):
        """Initialize an attribute feature extractor for retrieving all values for a specific attribute.
                
        Args:
            attribute_level: 'trace' or 'event'. Whether the attribute is on trace or event level.
            attribute_name: Name of the attribute.
        """
        super().__init__(f'Attribute: {attribute_name}')
        self.attribute_level = attribute_level
        self.attribute_name = attribute_name
            
    def extract(self, log):
        """Extract the attribute values from the given log.
        
        Args:
            log: A pm4py event log.
        
        Returns:
            Series of observations for the specified event log.
        """
        result_list = []
        
        if self.attribute_level == 'trace':
            for trace in log:
                result_list.append(trace.attributes[self.attribute_name])
        
        # convert to numpy array
        result_array = np.array(result_list)
        
        return result_array

class RelationalEntropyFeatureExtractor(FeatureExtractor):
    """Feature extractor that extracts relational entropy for each activity as a feature.
    """
    def __init__(self, direction='followed_by', activity_name_field='concept:name'):
        """Initialize a feature extractor that extracts the relational entropy feature.
        
        Args:
            direction: "followed_by" or "preceded_by". Direction of causality.
            activity_name_field: Field name in the event log to identify the activity name.
        """
        super().__init__('Relational Entropy')
        self.direction = direction
        self.activity_name_field = activity_name_field
    
    def extract(self, log):
        """Extract the relatinoal entropy for each activity in the given log.
        
        Args:
            log: A pm4py event log.
        
        Returns:
            The relational entropy for each activity in the log.
        """
        return features.get_relational_entropy(log, direction=self.direction, activity_name_field=self.activity_name_field)

    
def get_all_trace_attributes(log):
    """Get all trace-level attributes in a pm4py event log.
    
    Args:
        log: pm4py event log.
    
    Returns: Set of trace-level attributes in the event log.
    """
    attribute_set = set()
    for trace in log:
        attribute_set.update(trace.attributes)
    
    return attribute_set
