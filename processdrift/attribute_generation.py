"""Module for synthetically generating drifting attribute data to add to an event log.

The AttributeGenerator class generates categorical data attributes which either drift or don't. 

Use the `create_and_populate_attribute_generator()` function as a factory method for producing pre-configured attribute generators.
"""

import numpy as np
from opyenxes.model import XAttributeLiteral
from scipy import stats

RECURRING_DRIFT_MEAN_LENGTH = 20
RECURRING_DRIFT_STANDARD_DEVIATION = 4


class AttributeGenerator:
    """Simulates attributes for a given opyenxes event log

    TODO Implement continuous attribute generation. So far only categorical attributes are supported.
    TODO Implement incremental and gradual drift.
    TODO Implement event attribute generation. So far only trace attributes can be generated.
    """

    def __init__(self, opyenxes_log):
        """Initialize an attribute generator which can be used to generate multiple attributes for a given event log.

        Args:
            opyenxes_log: opyenxes event log to add attribute data to.
        """
        self.opyenxes_log = opyenxes_log

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
            drift_type: 'sudden' or 'recurring'. Type of drift, e.g., sudden.

        Returns:
            The combined data.
        """

        combined_data = baseline_data

        # How the baseline and drift data are combined depends on the drift type
        if drift_type == 'sudden':
            # For sudden drift, the drift data replaces the baseline data starting from the drift point.
            combined_data[drift_point:] = drifted_data[drift_point:]
        elif drift_type == 'recurring':
            # In case of recurring drift, the data drifts for a certain amount of time but then returns back to the base data

            # calculate the drift duration
            drift_duration = int(np.random.normal(
                RECURRING_DRIFT_MEAN_LENGTH, RECURRING_DRIFT_STANDARD_DEVIATION))
            if drift_duration < 1:
                drift_duration == 1

            combined_data[drift_point:drift_point +
                          drift_duration] = drifted_data[drift_point:drift_point+drift_duration]
        # TODO implement further types of drift

        return combined_data

    def _generate_categorical_data(self, distribution):
        """Generate categorical data for the length of the event log according to a distribution.

        The distribution array determines how many attributes are generated and what their sampling percentage is.

        Args:
            distribution: Python array with sampling distribution. E.g., [0.1, 0.5, 0.4].

        Returns:
            Array of generated attribute value sequence.        
        """
        attribute_value_candidates = [
            f'value_{attribute_number + 1}' for attribute_number, p in enumerate(distribution)]

        # get the data according to the probability distribution
        attribute_values = np.random.choice(attribute_value_candidates, len(
            self.opyenxes_log), p=distribution)  # one entry for each trace or event

        # convert to list
        attribute_values = list(attribute_values)

        return attribute_values

    def _write_attribute_into_log(self, attribute_name, attribute_data):
        """Write an attribute value sequence into a log.

        Args:
            attribute_name: Name of the attribute.
            attribute_data: Pre-generated attribute value sequence.
        """

        for trace, attribute_value in zip(self.opyenxes_log, attribute_data):
            # build the new attribute
            attribute = XAttributeLiteral.XAttributeLiteral(
                key=attribute_name, value=attribute_value)

            # add the new attribute to the existing trace attributes dictionary
            trace_attributes = trace.get_attributes()
            trace_attributes[attribute_name] = attribute

            # update the trace attribute dictionary
            trace.set_attributes(trace_attributes)

    def add_categorical_attribute(self,
                                  attribute_name,
                                  distribution):
        """Generate a categorical variable according to a given distribution.

        Args:
            attribute_name: Name of the attribute that is generated.
            distribution: Array with probabilities for each categorical value.
        """
        # generate the data
        attribute_data = self._generate_categorical_data(distribution)

        # write the new data into the log
        self._write_attribute_into_log(
            attribute_name, attribute_data)

    def add_drifting_categorical_attribute(self,
                                           attribute_name,
                                           base_distribution,
                                           drifted_distribution,
                                           explain_change_point,
                                           drift_type='sudden',
                                           standard_deviation_offset_explain_change_point=0):
        """Generates a categorical attribute that exhibits drift behavior at a certain drift point.

        Args:
            attribute_name: Name of the attribute that is generated.
            base_distribution: Array with probabilities for each categorical value. Serves as the baseline.
            drifted_distribution: Array with probabilities for each categorical value. Used to generate the the drifted data.
            explain_change_point: Which change point to explain.
            drift_type: 'sudden' or 'recurring'. The type of data drift.   
            sd_offset_explain_change_point: How far the attribute drift shall be offsetted from the explainable change point. Measured in standard deviation of a normal distribution.
        """

        # generate the baseline data
        baseline_data = self._generate_categorical_data(base_distribution)

        # generate the changed data
        drifted_data = self._generate_categorical_data(drifted_distribution)

        # get change point of categorical attribute
        mean = 0
        sd = standard_deviation_offset_explain_change_point
        offset_from_explainable_change_point = int(
            abs(np.random.normal(loc=mean, scale=sd)))
        change_point = explain_change_point - offset_from_explainable_change_point

        # combine the baseline and drifted data
        combined_data = self._combine_concepts(
            baseline_data, change_point, drifted_data, drift_type)

        # write the new data into the log
        self._write_attribute_into_log(
            attribute_name, combined_data)

        # put information about this change point into the change_point_explanations array
        change_point_info = {}
        change_point_info['attribute_name'] = attribute_name
        change_point_info['base_distribution'] = base_distribution
        change_point_info['explain_change_point'] = explain_change_point
        change_point_info['change_point'] = change_point
        change_point_info['drift_type'] = drift_type

        self.change_point_explanations.append(change_point_info)


def _get_distribution(attribute_value_count, min_probability=0.05):
    """Get a single distribution for a given number of values.

    Args:
        number_values: How many different values the distribution should have.
        min_probability: Minimum probability per attribute value.

    Returns:
        A probability distribution.
    """
    # generate a random number per attribute_value_count
    # each attribute probability should be at least min_percent -> return an error if this cannot be satisfied
    if attribute_value_count * min_probability > 1:
        raise Exception(
            "'min_probability' too high for the number of attribute values.")

    sampled_random_numbers = np.random.random(attribute_value_count)
    distribution_random = sampled_random_numbers / sum(sampled_random_numbers)
    distribution = [min_probability + (1-(min_probability*attribute_value_count))
                    * random for random in distribution_random]

    return distribution


def _get_drifted_distributions(attribute_value_count, change_type=None, min_hellinger_distance=0.3, max_tries = 100):
    """Get two distributions, the baseline distribution and the drifted distribution.

    The change_type determines in which regard both are different.

    Args:
        attribute_value_count: How many attribute values there are.
        change_type: 'new_value', 'new_distribution'. New value introduces a new value in the changed distribution. New distribution completely changes the new distribution.
        min_hellinger_distance: Minimum Hellinger distance between baseline and drifted distribution. Only applicable for the 'new_distribution' change type.

    Returns:
        (baseline_distribution, drifted_distribution) tuple
    """
    if attribute_value_count < 2:
        raise Exception('Must generate at least 2 attribute values.')

    if change_type is None:
        change_type = np.random.choice(
            ['new_value', 'new_distribution'])

    baseline_distribution = None
    drifted_distribution = None
    # implement different strategies per change type

    if change_type == 'new_value':
        baseline_distribution = _get_distribution(attribute_value_count - 1)
        # add a 0% probability item
        baseline_distribution = _get_distribution(attribute_value_count - 1)
        baseline_distribution.append(0)

        new_value_probability = np.random.random()

        drifted_distribution = baseline_distribution.copy()
        drifted_distribution = list(np.array(drifted_distribution) * (1 - new_value_probability))
        drifted_distribution[-1] = new_value_probability
    elif change_type == 'new_distribution':
        # get the baseline distribution
        baseline_distribution = None
        if change_type == 'new_value':
            baseline_distribution = _get_distribution(attribute_value_count - 1)
            # add a 0% probability item
            baseline_distribution.append(0)
        else:
            baseline_distribution = _get_distribution(attribute_value_count)
            trial_number = 0

            while ((trial_number < max_tries) and (drifted_distribution is None)):            
                drifted_distribution_candidate = _get_distribution(attribute_value_count)
                hellinger_distance = np.sqrt(np.sum((np.sqrt(baseline_distribution) - np.sqrt(drifted_distribution_candidate)) ** 2)) / np.sqrt(2)
                
                if hellinger_distance <= min_hellinger_distance:
                    drifted_distribution = drifted_distribution_candidate
                
                trial_number += 1
    
        if not drifted_distribution:
            raise Exception("Could not find a drifting distribution at the specified minimum hellinger distance.")

    return baseline_distribution, drifted_distribution


def create_and_populate_attribute_generator(event_log,
                                            change_points,
                                            count_relevant_attributes,
                                            count_irrelevant_attributes,
                                            number_attribute_values=3,
                                            type_of_drift='sudden',
                                            type_of_change='mixed',
                                            standard_deviation_offset_explain_change_point=0,
                                            min_hellinger_distance = 0.3,
                                            max_tries = 100
                                            ):
    """Given an pm4py event log, add synthetic attribute data.
    """
    ag = AttributeGenerator(event_log)

    # check that the number of relevant attributes does not exceed the number of changepoints
    if count_relevant_attributes > len(change_points):
        raise Exception(
            f"Not enough relevant attributes for number of change points. {count_relevant_attributes} relevant attributes for {len(change_points)} change points.")

    # generate drifted attributes
    for attribute_index in range(count_relevant_attributes):
        attribute_name = f'relevant_attribute_{(attribute_index + 1):02d}'

        # relevant_attribute_011 explains cp1...

        # get the distributions
        # if type of change is set to 'mixed', choose the change type
        this_change_type = None
        if type_of_change is None or type_of_change == 'mixed':
            this_change_type = np.random.choice(
                ['new_value', 'new_distribution'])
        else:
            this_change_type = type_of_change

        # generate the base and drifted distribution
        base_distribution, drifted_distribution = _get_drifted_distributions(number_attribute_values,
                                                                             change_type=this_change_type,
                                                                             min_hellinger_distance=min_hellinger_distance,
                                                                             max_tries=max_tries)

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
