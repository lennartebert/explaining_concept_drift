"""Module for synthetically generating drifting attribute data to add to an event log.

The AttributeGenerator class generates categorical data attributes which either drift or don't.

Use the `create_and_populate_attribute_generator()` function as a factory method for producing pre-configured attribute generators.
"""

import os

import numpy as np
from opyenxes.data_in import XesXmlParser
from opyenxes.data_out import XesXmlSerializer
from opyenxes.model import XAttributeLiteral
from scipy import stats

# mean length for recurring drift
RECURRING_DRIFT_MEAN_LENGTH = 20

# standard deviation for length of recurring drift
RECURRING_DRIFT_STANDARD_DEVIATION = 4

# maximum number of tries in which the minimum Hellinger distance between generated distributions needs to be achieved
MAX_TRIES_HELLINGER_DISTANCE = 100


class AttributeGenerator:
    """Simulates attributes for a given opyenxes event log

    TODO Implement continuous attribute generation. So far only categorical attributes are supported.
    TODO Implement incremental and gradual drift.
    TODO Implement event attribute generation. So far only trace attributes can be generated.
    """

    def __init__(self):
        """Initialize an attribute generator which can be used to generate multiple attributes for a given event log.
        """
        self.attributes_to_generate = []
        self.change_point_explanations = []

    class CategoricalAttribute:
        def __init__(self, name, distribution, drift=None):
            """Defines a categorical attribute.

            Args:
                name: Name of the attribute. Will be written to the event log.
                distribution: Array with probabilities for each categorical value. Serves as the baseline distribution
            """
            self.name = name
            self.distribution = distribution
            self.drift = drift

    class CategoricalDrift:
        def __init__(self,
                     distribution,
                     explain_change_point,
                     drift_type='sudden',
                     sd_offset_explain_change_point=0):
            """Defines a categorical drift

            Args:
                distribution: Array with probabilities for each categorical value. Used to generate the the drifted data.
                explain_change_point: Which change point to explain.
                type: 'sudden' or 'recurring'. The type of data drift.
                sd_offset_explain_change_point: How far the attribute drift shall be offsetted from the explainable change point. Measured in standard deviations.
            """
            self.distribution = distribution
            self.explain_change_point = explain_change_point
            self.drift_type = drift_type
            self.sd_offset_explain_change_point = sd_offset_explain_change_point

    def add_attribute(self, attribute):
        """Add an attribute which is to be generated.

        Args:
            attribute: An attribute. E.g., of class CategoricalAttribute."""
        self.attributes_to_generate.append(attribute)

    def generate(self, opyenxes_event_log):
        """Generate the synthetic attribute data.

        Args:
            opyenxes_event_log: opyenxes event log to which the attribute data is added.

        Returns:
            opyenxes event log with drifting attribute data.
            """

        # create a deep copy of the event log
        opyenxes_event_log = opyenxes_event_log.clone()
        
        length = len(opyenxes_event_log)

        for attribute in self.attributes_to_generate:
            attribute_data = None
            if isinstance(attribute, AttributeGenerator.CategoricalAttribute):
                # generate the data
                if attribute.drift is None:
                    attribute_data = self._generate_categorical_data(
                        length, attribute.distribution)
                else:
                    attribute_data = self._generate_drifting_categorical_attribute(
                        length, attribute, attribute.drift)

            # write attribute data to the event log
            self._write_attribute_into_log(
                opyenxes_event_log, attribute.name, attribute_data)

        return opyenxes_event_log

    def _generate_categorical_data(self, length, distribution):
        """Generate categorical data for the length of the event log according to a distribution.

        The distribution array determines how many attributes are generated and what their sampling percentage is.

        Args:
            length: Length of generated attribute value series.
            distribution: Python array with sampling distribution. E.g., [0.1, 0.5, 0.4].

        Returns:
            Array of generated attribute value sequence.        
        """
        attribute_value_candidates = [
            f'value_{attribute_number + 1}' for attribute_number, p in enumerate(distribution)]

        # get the data according to the probability distribution
        # one entry for each trace or event
        attribute_values = np.random.choice(
            attribute_value_candidates, length, p=distribution)

        # convert to list
        attribute_values = list(attribute_values)

        return attribute_values

    def _generate_drifting_categorical_attribute(self, length, attribute, drift):
        """Generates a categorical attribute that exhibits drift behavior at a certain drift point.

        Also writes the drift to the change point explanation list.

        Args:
            length: Length of generated attribute value series.
            attribute: The attribute which drifts. E.g., of class CategoricalAttribute.
            drift: The drifting behavior. E.g., of class CategoricalDrift.

        Returns:
            Array of generated attribute value sequence.    
        """

        # generate the baseline data
        baseline_data = self._generate_categorical_data(
            length, attribute.distribution)

        # generate the changed data
        drifted_data = self._generate_categorical_data(
            length, drift.distribution)

        # get change point of categorical attribute
        mean = 0
        sd = drift.sd_offset_explain_change_point
        offset_from_explainable_change_point = int(
            abs(np.random.normal(loc=mean, scale=sd)))
        change_point = drift.explain_change_point - offset_from_explainable_change_point

        # combine the baseline and drifted data
        combined_data = self._combine_concepts(
            baseline_data, change_point, drifted_data, drift.drift_type)

        # put information about this change point into the change_point_explanations array
        change_point_info = {}
        change_point_info['attribute_name'] = attribute.name
        change_point_info['base_distribution'] = attribute.distribution
        change_point_info['explain_change_point'] = drift.explain_change_point
        change_point_info['change_point'] = change_point
        change_point_info['drift_type'] = drift.drift_type

        self.change_point_explanations.append(change_point_info)

        return combined_data

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

    def _write_attribute_into_log(self, opyenxes_log, attribute_name, attribute_data):
        """Write an attribute value sequence into a log.

        Args:
            opyenxes_log: opyenxes event log to write attributes into.
            attribute_name: Name of the attribute.
            attribute_data: Pre-generated attribute value sequence.
        """

        for trace, attribute_value in zip(opyenxes_log, attribute_data):
            # build the new attribute
            attribute = XAttributeLiteral.XAttributeLiteral(
                key=attribute_name, value=attribute_value)

            # add the new attribute to the existing trace attributes dictionary
            trace_attributes = trace.get_attributes()
            trace_attributes[attribute_name] = attribute

            # update the trace attribute dictionary
            trace.set_attributes(trace_attributes)


def _get_distribution(attribute_value_count):
    """Get a single distribution for a given number of values.

    Args:
        attribute_value_count: How many different values the distribution should have.

    Returns:
        A probability distribution.
    """
    # generate a random number per attribute_value_count

    sampled_random_numbers = np.random.random(attribute_value_count)
    distribution = sampled_random_numbers / sum(sampled_random_numbers)

    return distribution


def _get_drifted_distributions(attribute_value_count, change_type=None, min_hellinger_distance=0.3):
    """Get two distributions, the baseline distribution and the drifted distribution.

    The change_type determines in which regard both are different.

    Args:
        attribute_value_count: How many attribute values there are.
        change_type: 'new_value', 'new_distribution'. 'new_value' introduces a new value in the changed distribution. 'new_distribution' completely changes the new distribution.
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
        baseline_distribution = np.append(baseline_distribution, 0)

        new_value_probability = np.random.random()

        drifted_distribution = baseline_distribution.copy()
        drifted_distribution = list(
            np.array(drifted_distribution) * (1 - new_value_probability))
        drifted_distribution[-1] = new_value_probability
    elif change_type == 'new_distribution':
        # get the baseline distribution
        baseline_distribution = _get_distribution(attribute_value_count)
        trial_number = 0

        while ((trial_number < MAX_TRIES_HELLINGER_DISTANCE) and (drifted_distribution is None)):
            drifted_distribution_candidate = _get_distribution(
                attribute_value_count)
            hellinger_distance = np.sqrt(np.sum((np.sqrt(
                baseline_distribution) - np.sqrt(drifted_distribution_candidate)) ** 2)) / np.sqrt(2)

            if min_hellinger_distance is None or hellinger_distance >= min_hellinger_distance:
                drifted_distribution = drifted_distribution_candidate

            trial_number += 1

        if drifted_distribution is None:
            raise Exception(
                "Could not find a drifting distribution at the specified minimum hellinger distance.")

    return baseline_distribution, drifted_distribution


def create_and_populate_attribute_generator(change_points,
                                            count_relevant_attributes,
                                            count_irrelevant_attributes,
                                            number_attribute_values=3,
                                            drift_type='sudden',
                                            distribution_change_type='mixed',
                                            sd_offset_explain_change_point=0,
                                            min_hellinger_distance=0.3
                                            ):
    """Factory function for the AttributeGenerator class.

    Pre-populates an attribute generator with drifting and not drifting attributes.

    Args:
        change_points: Change points at which attributes may drift.
        count_relevant_attributes: Number of drifting attributes.
        count_irrelevant_attributes: Number of not-drifting attributes.
        number_attribute_values: Number of values per attribute.
        drift_type: 'sudden' or 'recurring'; The type of drift.
        distribution_change_type: 'new_value', 'new_distribution' or 'mixed'; How the value distribution changes if a drift occurs. 'new_value' introduces a new value in the changed distribution. 'new_distribution' completely changes the new distribution. 'mixed' selects randomly from the two.
        sd_offset_explain_change_point: Standard deviation of attribute change to change point.
        min_hellinger_distance: Minimum Hellinger distance between drifting and baseline distributions.
    """
    ag = AttributeGenerator()

    # generate drifted attributes
    for attribute_index in range(count_relevant_attributes):
        attribute_name = f'relevant_attribute_{(attribute_index + 1):02d}'

        # relevant_attribute_011 explains cp1...

        # get the distributions
        # if type of change is set to 'mixed', choose the change type
        this_change_type = None
        if distribution_change_type is None or distribution_change_type == 'mixed':
            this_change_type = np.random.choice(
                ['new_value', 'new_distribution'])
        else:
            this_change_type = distribution_change_type

        # generate the base and drifted distribution
        base_distribution, drifted_distribution = _get_drifted_distributions(number_attribute_values,
                                                                             change_type=this_change_type,
                                                                             min_hellinger_distance=min_hellinger_distance
                                                                             )

        # get change point to explain
        explain_change_point = change_points[attribute_index % len(
            change_points)]

        drift = AttributeGenerator.CategoricalDrift(
            drifted_distribution, explain_change_point, drift_type, sd_offset_explain_change_point)

        attribute = AttributeGenerator.CategoricalAttribute(
            attribute_name, base_distribution, drift)

        ag.add_attribute(attribute)

    # generate attributes that do not drift
    for attribute_index in range(count_irrelevant_attributes):
        attribute_name = f'irrelevant_attribute_{(attribute_index + 1):02d}'
        distribution = _get_distribution(number_attribute_values)

        attribute = AttributeGenerator.CategoricalAttribute(
            attribute_name, base_distribution)

        ag.add_attribute(attribute)

    # return the attribute generator
    return ag


def apply_attribute_generator_to_log_file(attribute_generator, in_file_path, out_file_path):
    """Read an XES log file, generate attributes, and write the event log with augmented attributes.

    Attributes:
        attribute_generator: A prepared attribute generator.
        in_file_path: File path to the .XES event log file.
        out_file_path: Output path for the .XES event log file with added attributes.
    """
    # read the XES event log
    opyenxes_xes_parser = XesXmlParser.XesXmlParser()
    opyenxes_logs = None

    with open(in_file_path) as data_file:
        opyenxes_logs = opyenxes_xes_parser.parse(data_file)

    # raise an exception if there are multiple logs
    if len(opyenxes_logs) > 1:
        raise Exception("More than one event log in the XES file.")

    original_log = opyenxes_logs[0]

    # apply the attribute generator
    generated_log = attribute_generator.generate(original_log)

    # write the result to the output file
    opyenxes_xes_serializer = XesXmlSerializer.XesXmlSerializer()

    # create path if doesn't exist
    if not os.path.exists(os.path.dirname(out_file_path)):
        # Create a new directory because it does not exist
        os.makedirs(os.path.dirname(out_file_path))

    with open(out_file_path, 'w') as data_file:
        opyenxes_xes_serializer.serialize(generated_log, data_file)
