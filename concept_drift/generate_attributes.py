from opyenxes.model import XAttributeLiteral
import numpy as np

REOCCURING_DRIFT_MEAN_LENGTH = 20
REOCCURING_DRIFT_STANDARD_DEVIATION = 4

class AttributeGenerator:
    """Simulates attributes for a given opyenxes event log"""
    def __init__(self, opyenxes_log, change_points, timestamp_field = 'time:timestamp'):
        """Initialize an attribute generator which can be used to generate multiple attributes for a given event log.
        
        Args:
            opyenxes_log: opyenxes event log
            change_points: List of change points
            timestamp filed: Field name of event timestamp field
        """
        self.opyenxes_log = opyenxes_log
        self.change_points = change_points
        
        # get start and end trace
        self.start_trace_id = 0
        self.end_trace_id = len(opyenxes_log) - 1
        
        # get start and end timestamp
        self.start_time = opyenxes_log[0][0].get_attributes()['time:timestamp'].get_value()
        
        #iterate through all traces to find the last timestamp
        self.end_time = opyenxes_log[-1][-1].get_attributes()['time:timestamp'].get_value()
        for trace in reversed(opyenxes_log):
            trace_end_time = trace[-1].get_attributes()['time:timestamp'].get_value()
            if trace_end_time > self.end_time:
                self.end_time = trace_end_time
                
        # create a dictionary that holds information about which attribute drift explains which change point
        self.change_point_explanations = {}
        for change_point in self.change_points:
            self.change_point_explanations[change_point] = []
    
    def _combine_concepts(self, baseline_data, drift_points, drift_data, drift_type):
        """Combine different concepts into one data stream.
        
        Args:
            baseline_data: The baseline data (original concept).
            drift_points: List of points when concept drift occurs.
            drift_data: List of drifted data (one for each drift point).
            drift_type: 'sudden' or 'gradual'. Type of drift, e.g., sudden.
        """
        # perform some integrity checks
        if not len(drift_points) == len(drift_data): raise Exception('Number of drift points and number of drift data sets need to be the same.')
        if drift_type not in ['sudden']: raise Exception(f'Drit type {drift_type} not implemented')
        
        # make sure the drift points are ordered
        drift_point_data_tuples = []
        for drift_point, drift_data_instance in zip(drift_points, drift_data):
            drift_point_data_tuples.append((drift_point, drift_data_instance))
        drift_point_data_tuples.sort(key=lambda tup: tup[0])        
        
        attribute_value_series = baseline_data
        # implement sudden drift
        if drift_type == 'sudden':
            for drift_point, drift_data_instance in drift_point_data_tuples:
                # place the changed attribute data starting from the changepoint
                attribute_value_series[drift_point:] = drift_data_instance[drift_point:]
        elif drift_type == 'reoccurring':
            # calculate the drift duration
            drift_duration = int(np.random.normal(REOCCURING_DRIFT_MEAN_LENGTH, REOCCURING_DRIFT_STANDARD_DEVIATION))
            if drift_duration < 1: drift_duration == 1
            
            attribute_value_series[drift_point:drift_point+drift_duration] = drift_data_instance[drift_point:drift_point+drift_duration]
        # TODO implement further types of drift
        
        return attribute_value_series
    
    def _get_process_to_attribute_changepoint_mapping(self, explain_change_points, change_location_standard_deviation):
        """Get the location of each attribute change point.
        
        Args:
            explain_change_points: Changepoints to explain.
            change_location_standard_deviation: Standard deviation of the change location around the process change point. Attribute change points always occur prior to the process change point.
        """
        
        # get the location of the attribute change points for each of the sampled change points
        attribute_change_points = []
        for change_point in explain_change_points:
            # calculate the attribute change point deviation
            attribute_change_point_deviation = np.abs(np.random.normal(0, change_location_standard_deviation))
            attribute_change_point = int(change_point - attribute_change_point_deviation)
            if attribute_change_point < 0: # handle the case that we try to get a non-existing trace
                attribute_change_point == 0
            
            attribute_change_points.append(attribute_change_point)
        
        return explain_change_points, attribute_change_points
    
    def _generate_attribute(self):
        pass
    
    def generate_continuous_attribute(self, attribute_name, baseline_distribution,
                                      baseline_noise, explain_change_points=None,
                                       change_location_standard_deviation=10,
                                       drift_type='sudden', attribute_level='trace', 
                                       noise_level=0, concept_change='oversampling', 
                                       data_stationarity='stationary'):
        
        pass
    
    def _generate_categorical_data(self, count_attribute_values):
        attribute_value_candidates = [f'value_{attribute_number + 1}' for attribute_number in range(count_attribute_values)]
        
        # assing a static percentage of category occurences
        # use the Dirichlet distribution
        probabilities = np.random.dirichlet(np.ones(count_attribute_values))
        
        # get the data according to the probability distribution
        attribute_values = [] # one entry for each trace or event
        for trace in range(len(self.opyenxes_log)):
            attribute_value = np.random.choice(attribute_value_candidates, p=probabilities)
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
    
    def generate_categorical_attribute(self, attribute_name, count_attribute_values=3,
                                       explain_change_points=None, 
                                       change_location_standard_deviation=10,
                                       drift_type='sudden', attribute_level='trace', 
                                       noise_level=0, concept_change='oversampling', 
                                       data_stationarity='stationary'):
        # if explain_change_points is set to None, explain all no change points
        
        # generate the baseline data
        attribute_data = self._generate_categorical_data(count_attribute_values)
        
        if explain_change_points != None:
            # generate the changed attribute data
            drift_data = []
            for attribute_change in range(len(explain_change_points)):
                new_drift_data = self._generate_categorical_data(count_attribute_values)
                drift_data.append(new_drift_data)
            
            # determine the attribute change points
            explain_change_points, attribute_change_points = self._get_process_to_attribute_changepoint_mapping(explain_change_points, change_location_standard_deviation)

            # combine the baseline and changed attribute data at the change points
            attribute_data = self._combine_concepts(attribute_data, attribute_change_points, drift_data, drift_type)
              
            # add changepoint explanation to explanation dataframe
            for explain_change_point, attribute_change_point in zip(explain_change_points, attribute_change_points):
                change_point_info = {}
                change_point_info['attribute_name'] = attribute_name
                change_point_info['drift_type'] = drift_type
                change_point_info['drift_location'] = attribute_change_point
                self.change_point_explanations[explain_change_point].append(change_point_info)
        
        # add categorical noise by replacing x% of values with yet another distribution
        if noise_level != 0:
            # sample a new distribution
            noise_data = self._generate_categorical_data(count_attribute_values)
            
            replace_values = np.random.choice([False, True], size=len(attribute_data), replace=True, p=[1-noise_level, noise_level])
            
            attribute_data_np = np.array(attribute_data)
            noise_data_np = np.array(noise_data)
                                     
            attribute_data_np[replace_values] = noise_data_np[replace_values]
            
            # convert back to standard python array
            attribute_data = attribute_data_np.tolist()
        
        # write change into log
        self._write_attribute_into_log(attribute_name, attribute_data, attribute_level)

        # return change point information
        return self.change_point_explanations