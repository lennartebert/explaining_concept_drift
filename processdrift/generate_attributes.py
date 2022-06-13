from opyenxes.model import XAttributeLiteral
import numpy as np
from scipy import stats

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
            drift_type: 'sudden' or 'reoccurring'. Type of drift, e.g., sudden.
        
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
    
    def add_categorical_attribute(self,
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
        
    def add_drifting_categorical_attribute(self, 
                                       attribute_name, 
                                       base_distribution,
                                       drifted_distribution,
                                       explain_change_point,
                                       drift_type = 'sudden',
                                       attribute_level = 'trace',
                                       standard_deviation_offset_explain_change_point = 0):
        """Generates a categorical attribute that exhibits drift behavior at a certain drift point.
        
        Args:
            attribute_name: Name of the attribute that is generated.
            base_distribution: Array with probabilities for each categorical value. Serves as the baseline.
            drifted_distribution: Array with probabilities for each categorical value. Used to generate the the drifted data.
            explain_change_point: Which change point to explain.
            drift_type: 'sudden' or 'reoccurring'. The type of data drift.   
            sd_offset_explain_change_point: How far the attribute drift shall be offsetted from the explainable change point. Measured in standard deviation of a normal distribution.
        """

        # generate the baseline data
        baseline_data = self._generate_categorical_data(base_distribution)
        
        # generate the changed data
        drifted_data = self._generate_categorical_data(drifted_distribution)
        
        # get change point of categorical attribute
        mean = 0
        sd = standard_deviation_offset_explain_change_point
        offset_from_explainable_change_point = int(abs(np.random.normal(loc=mean, scale=sd)))
        change_point = explain_change_point - offset_from_explainable_change_point

        # combine the baseline and drifted data
        combined_data = self._combine_concepts(baseline_data, change_point, drifted_data, drift_type)
        
        # write the new data into the log
        self._write_attribute_into_log(attribute_name, combined_data, attribute_level)
        
        # put information about this change point into the change_point_explanations array
        change_point_info = {}
        change_point_info['attribute_name'] = attribute_name
        change_point_info['base_distribution'] = base_distribution
        change_point_info['explain_change_point'] = explain_change_point
        change_point_info['change_point'] = change_point
        change_point_info['drift_type'] = drift_type
        
        self.change_point_explanations.append(change_point_info)
    

def _get_distribution(attribute_value_count):
    """Get a single distribution for a given number of values.
    
    Args:
        number_values: How many different values the distribution should have.
    
    Returns:
        A probability distribution.
    """
    # use the Dirichlet distribution
    distribution = np.random.dirichlet(np.ones(attribute_value_count)).tolist()
    return distribution


def _get_drifted_distributions(attribute_value_count, change_type=None):
    """Get two distributions, the baseline distribution and the drifted distribution.
    
    The change_type determines in which regard both are different.
    
    The two distributions are guaranteed to be significantly different at 10 observations.
    
    Args:
        attribute_value_count: How many attribute values there are.
        change_type: 'new_value', 'new_distribution'. New value introduces a new value in the changed distribution. New distribution completely changes the new distribution.
    
    Returns:
        (baseline_distribution, drifted_distribution) tuple
    """
    if attribute_value_count < 2: raise Exception('Must generate at least 2 attribute values.')
    
    if change_type is None:
        change_type = np.random.choice(['new_value', 'overproportional_gain', 'independent_new'])
    
    # get the baseline distribution
    baseline_distribution = None
    if change_type == 'new_value':
        baseline_distribution = _get_distribution(attribute_value_count - 1)
        # add a 0% probability item
        baseline_distribution.append(0)
    else:
        baseline_distribution = _get_distribution(attribute_value_count)
        
    drifted_distribution_found = False
    drifted_distribution = None
    
    while not drifted_distribution_found:
        if change_type == 'new_value':
            new_value_probability = np.random.random()

            drifted_distribution = baseline_distribution.copy()
            drifted_distribution = list(np.array(drifted_distribution) * (1 - new_value_probability))
            drifted_distribution[-1] = new_value_probability
        else:
            drifted_distribution = _get_distribution(attribute_value_count)
        
        hellinger_distance = np.sqrt(np.sum((np.sqrt(baseline_distribution) - np.sqrt(drifted_distribution)) ** 2)) / np.sqrt(2)
        
        if hellinger_distance > 0.3:
            drifted_distribution_found = True

    return baseline_distribution, drifted_distribution


def create_and_populate_attribute_generator(event_log, 
    change_points,
    count_relevant_attributes,
    count_irrelevant_attributes,
    number_attribute_values=3,
    type_of_drift='sudden',
    type_of_change='mixed',
    standard_deviation_offset_explain_change_point=0):
    """Given an pm4py event log, add synthetic attribute data.
    """
    ag = AttributeGenerator(event_log, change_points)


    # check that the number of relevant attributes does not exceed the number of changepoints
    if count_relevant_attributes > len(change_points):
        raise Exception(f"Not enough relevant attributes for number of change points. {count_relevant_attributes} relevant attributes for {len(change_points)} change points.")

    # generate drifted attributes
    for attribute_index in range(count_relevant_attributes):
        attribute_name = f'relevant_attribute_{(attribute_index + 1):02d}'
        
        # relevant_attribute_011 explains cp1...

        # get the distributions
        # if type of change is set to 'mixed', choose the change type
        this_change_type = None
        if type_of_change is None or type_of_change == 'mixed':
            this_change_type = np.random.choice(['new_value', 'new_distribution'])
        else:
            this_change_type = type_of_change

        # generate the base and drifted distribution
        base_distribution, drifted_distribution = _get_drifted_distributions(number_attribute_values, 
                                                                        change_type=this_change_type)

        # get change point to explain
        explain_cp = change_points[attribute_index]
            
        ag.add_drifting_categorical_attribute(attribute_name,
                                        base_distribution,
                                        drifted_distribution=drifted_distribution,
                                        explain_change_point=explain_cp,
                                        drift_type=type_of_drift,
                                        standard_deviation_offset_explain_change_point=standard_deviation_offset_explain_change_point)
    
    # generate attributes that do not drift
    for attribute_index in range(count_irrelevant_attributes):
        attribute_name = f'irrelevant_attribute_{(attribute_index + 1):02d}'
        distribution = _get_distribution(number_attribute_values)
        ag.add_categorical_attribute(attribute_name, distribution)
    
    # return the attribute_generator
    return ag
