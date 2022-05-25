from opyenxes.model import XAttributeLiteral
import numpy as np

REOCCURING_DRIFT_MEAN_LENGTH = 20
REOCCURING_DRIFT_STANDARD_DEVIATION = 4

class AttributeGenerator:
    """Simulates attributes for a given opyenxes event log
    
    TODO implement continuous attribute generation.
    """
    def __init__(self, opyenxes_log, change_points):
        """Initialize an attribute generator which can be used to generate multiple attributes for a given event log.
        
        Args:
            opyenxes_log: opyenxes event log
            change_points: List of change points
        """
        self.opyenxes_log = opyenxes_log
        self.change_points = change_points
        
        # get start and end trace
        self.start_trace_id = 0
        self.end_trace_id = len(opyenxes_log) - 1
                
        # create a list that holds information about which attribute drift explains which change point
        self.change_point_explanations = []
    
    def _combine_concepts(self, baseline_data, drift_point, drifted_data, drift_type):
        """Combine different concepts into one data stream.
        
        Args:
            baseline_data: The baseline data (original concept).
            drift_point: Point of drift.
            drifted_data: Data for after the drift.
            drift_type: 'sudden' or 'gradual'. Type of drift, e.g., sudden.
        
        Returns:
            The combined data.
        """
        
        combined_data = baseline_data
        
        # How the baseline and drift data are combined depends on the drift type
        if drift_type == 'sudden':
            # For sudden drift, the drift data replaces the baseline data starting from the drift point.
            combined_data[drift_point:] = drifted_data[drift_point:]
        elif drift_type == 'reoccurring':
            # In case of reoccurring drift, the data drifts for a certain amount of time but then returns back to the base data
            
            # calculate the drift duration
            drift_duration = int(np.random.normal(REOCCURING_DRIFT_MEAN_LENGTH, REOCCURING_DRIFT_STANDARD_DEVIATION))
            if drift_duration < 1: drift_duration == 1
            
            combined_data[drift_point:drift_point+drift_duration] = drifted_data[drift_point:drift_point+drift_duration]
        # TODO implement further types of drift
        
        return combined_data
    
    def _generate_categorical_data(self, distribution):
        attribute_value_candidates = [f'value_{attribute_number + 1}' for attribute_number, p in enumerate(distribution)]
        
        # get the data according to the probability distribution
        attribute_values = [] # one entry for each trace or event
        for trace in range(len(self.opyenxes_log)):
            attribute_value = np.random.choice(attribute_value_candidates, p=distribution)
            attribute_values.append(attribute_value)
        
        return attribute_values
    
    def _write_attribute_into_log(self, attribute_name, attribute_data, attribute_level='trace'):
        """Write an attribute value sequence into a log.
        """
        for trace, attribute_value in zip(self.opyenxes_log, attribute_data):
            # build the new attribute
            attribute = XAttributeLiteral.XAttributeLiteral(key=attribute_name, value=attribute_value)
            
            # add the new attribute to the existing trace attributes dictionary
            trace_attributes = trace.get_attributes()
            trace_attributes[attribute_name] = attribute
            
            # update the trace attribute dictionary
            trace.set_attributes(trace_attributes)
    
    def generate_categorical_attribute(self,
                                       attribute_name,
                                       distribution, 
                                       attribute_level='trace'):
        """Generate a categorical variable according to a given distribution
        
        TODO: implement event level attribute generation
        
        Args:
            attribute_name: Name of the attribute that is generated.
            distribution: Array with probabilities for each categorical value.
            attribute_level: 'trace' or 'event'.
        """
        # generate the data
        attribute_data = self._generate_categorical_data(distribution)
        
        # write the new data into the log
        self._write_attribute_into_log(attribute_name, attribute_data, attribute_level)
        
    def generate_drifting_categorical_attribute(self, 
                                       attribute_name, 
                                       base_distribution,
                                       drifted_distribution,
                                       explain_change_point,
                                       drift_type = 'sudden',
                                       attribute_level = 'trace'):
        """Generates a categorical attribute that exhibits drift behavior at a certain drift point.
        
        Args:
            attribute_name: Name of the attribute that is generated.
            base_distribution: Array with probabilities for each categorical value. Serves as the baseline.
            drifted_distribution: Array with probabilities for each categorical value. Used to generate the the drifted data.
            explain_change_point: Which change point to explain.
            drift_type: 'sudden' or reoccurring. The type of data drift.            
        """
        # generate the baseline data
        baseline_data = self._generate_categorical_data(base_distribution)
        
        # generate the changed data
        drifted_data = self._generate_categorical_data(drifted_distribution)
        
        # combine the baseline and drifted data
        combined_data = self._combine_concepts(baseline_data, explain_change_point, drifted_data, drift_type)
        
        # write the new data into the log
        self._write_attribute_into_log(attribute_name, combined_data, attribute_level)
        
        # put information about this change point into the change_point_explanations array
        change_point_info = {}
        change_point_info['attribute_name'] = attribute_name
        change_point_info['base_distribution'] = base_distribution
        change_point_info['explain_change_point'] = explain_change_point
        change_point_info['drift_type'] = drift_type
        
        self.change_point_explanations.append(change_point_info)