"""Module with all classes and methods for feature extraction.
"""

from abc import ABC, abstractmethod
import numpy as np
import pm4py

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


class AttributeFE(FeatureExtractor):
    """The attribute feature extractor retrieves all attribute values from an event log.
    """
    def __init__(self, attribute_level, attribute_name):
        """Initialize an attribute feature extractor for retrieving all values for a specific attribute.
                
        Args:
            attribute_level: 'trace' or 'event'. Whether the attribute is on trace or event level.
            attribute_name: Name of the attribute.
            included_none: If the attribute is not set for the event/trace, include "None" in the result list.
        """
        super().__init__(f'{attribute_name}')
        self.attribute_level = attribute_level
        self.attribute_name = attribute_name
            
    def extract(self, event_log):
        """Extract the attribute values from the given log.
        
        Args:
            event_log: A pm4py event log.
        
        Returns:
            Series of observations for the specified event log.
        """
        result_list = []
        
        if self.attribute_level == 'trace':
            for trace in event_log:
                if self.attribute_name in trace.attributes:
                    result_list.append(trace.attributes[self.attribute_name])
                else:
                    result_list.append('Attribute not defined.')
        elif self.attribute_level == 'event':
            event_stream = pm4py.convert.convert_to_event_stream(event_log)
            for event in event_stream:
                if self.attribute_name in event:
                    result_list.append(event[self.attribute_name])
                else:
                    result_list.append('Attribute not defined.')
        
        # convert to numpy array
        result_array = np.array(result_list)
        
        return result_array

class RelationalEntropyFE(FeatureExtractor):
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

class RelationshipTypesCountFE(FeatureExtractor):
    def __init__(self):
        """Initialize a feature extractor that extracts the relationship types count feature.
        """
        super().__init__('Relationship Types Count')

    """Extracts the relationship type counts for each activity. Introduced by Bose et al. 2011
    """
    def extract(self, log):
        bi_directional_rtc = features.get_bi_directional_relationship_type_counts(log)
        return bi_directional_rtc

class RunsFE(FeatureExtractor):
    def __init__(self):
        """Initialize a feature extractor that extracts the runs feature.
        """
        super().__init__('Runs')

    """Extracts the number of runs in a given log. Introduced by Maaradji et al. 2015.
    """
    def extract(self, log):
        # get all traces that are in the log
        traces = features._get_traces(log)
        runs = features.get_runs(traces)

        return runs

def get_all_trace_attributes(log):
    """Get all trace-level attributes in a pm4py event log.
    
    Args:
        log: pm4py event log.
    
    Returns: Set of trace-level attributes in the event log.
    """
    attribute_set = pm4py.statistics.attributes.log.get.get_all_trace_attributes_from_log(log)
    
    return attribute_set

def get_all_event_attributes(log):
    """Get all event-level attributes in a pm4py event log.
    
    Args:
        log: pm4py event log.
    
    Returns: Set of event-level attributes in the event log.
    """
    attribute_set = pm4py.statistics.attributes.log.get.get_all_event_attributes_from_log(log)
    
    # needs to remove all trace/case attributes from the attribute set if they occur
    # case attributes always start with 'case:'
    case_attributes = [attribute for attribute in attribute_set if attribute.startswith('case:')]
    attribute_set.difference_update(case_attributes)

    return attribute_set
