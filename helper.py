# imports
import csv
import json
import math
import numbers
import os
import time
from operator import itemgetter

import numpy as np
import pandas as pd
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.util import xes_constants as xes

from processdrift import attribute_generation
from processdrift.explanation import (change_point_extraction, drift_detection,
                                      drift_explanation, evaluation,
                                      feature_extraction,
                                      population_comparison, windowing)


def create_and_get_new_path(old_file_path, old_base_path, new_base_path, new_extension=None):
    """Get a new file path. Also creates all folder for this path.

    Args:
        old_file_path: Previous path to file.
        old_base_path: Previous base path of data folder.
        new_base_path: New base directory.
        new_extension: Optional: New extension of file path. (E.g., .XES)

    Returns: New file path.
    """
    # get file path only from after the old_base_path
    rel_path = os.path.relpath(old_file_path, old_base_path)
    new_path = os.path.normpath(os.path.join(new_base_path, rel_path))

    if new_extension is not None:
        file_name, extension = os.path.splitext(new_path)
        new_path = file_name + new_extension

    # create path if not exists
    # create to_path if not existing
    if not os.path.exists(os.path.dirname(new_path)):
        # Create a new directory because it does not exist
        os.makedirs(os.path.dirname(new_path))

    # cleanup path
    new_path = os.path.normpath(new_path)

    return new_path


def get_change_points_maardji_et_al_2013(size):
    """Get the change point lists for the Maaradji et al. 2013 datasets.

    Args:
        size: Length of the event log. E.g., 10000.

    Returns:
        List of change points.
    """
    change_points = [int((i + 1)*size/10) for i in range(9)]
    return change_points


def get_attributes_and_types_for_synthetic_data(relevant_attributes=5, irrelevant_attributes=5):
    """Get attributes and types for the generated datasets.

    Args:
        relevant_attributes: How many relevant attributes are in the generated dataset?
        irrelevant_attributes: How many non-driften attributes are in the generated dataset?

    Returns:
        List of attribute name, level and type triplets.
    """
    list_of_triplets = []
    attribute_level = 'trace'
    attribute_type = 'categorical'
    for i in range(irrelevant_attributes):
        attribute_name = f'irrelevant_attribute_{(i + 1):02d}'
        triplet = (attribute_name, attribute_level, attribute_type)
        list_of_triplets.append(triplet)
    for i in range(relevant_attributes):
        attribute_name = f'relevant_attribute_{(i + 1):02d}'
        triplet = (attribute_name, attribute_level, attribute_type)
        list_of_triplets.append(triplet)
    return list_of_triplets


def _get_data_type(attribute_value):
    """Gets the datatype of an attribute based on the python type.

    Args:
        attribute_value: Value of an attribute.capitalize

    Returns:
        Whether the attribute is 'categorical' or 'continuous' based on its python type.
    """
    if isinstance(attribute_value, numbers.Number):
        return 'continuous'
    else:
        return 'categorical'


def automatically_get_attributes_and_data_types(event_log, selected_trace_attributes=None, selected_event_attributes=None):
    """Automatically gets a list of triplets of available attributes in an event log including their datatypes.

    The datatype of the attribute is determined by observing the datatypes in the event log.

    Args:
        event_log: A pm4py event log.
        selected_trace_attributes: Only get data types for the specified trace attributes. If None, all trace attributes are returned.
        selected_event_attributes: Only get data types for the specified event attributes. If None, all event attributes are returned.

    Returns:
        List of (attribute_value, attribute_level, attribute_type) triplets.
    """

    # build a dictionary with all attributes on trace and event level
    attributes_and_observed_types = {}

    for trace in event_log:
        # get the trace attribute types
        trace_attributes = trace.attributes

        # only keep those trace attributes that are in the trace attribute list

        for attribute_name, attribute_value in trace_attributes.items():
            if selected_trace_attributes is not None:
                if attribute_name not in selected_trace_attributes:
                    continue
            trace_attribute_tuple = (attribute_name, 'trace')
            if trace_attribute_tuple not in attributes_and_observed_types:
                attributes_and_observed_types[trace_attribute_tuple] = set(
                    [_get_data_type(attribute_value)])
            else:
                attributes_and_observed_types[trace_attribute_tuple].add(
                    _get_data_type(attribute_value))

        # get the event attribute types
        for event in trace:
            for attribute_name, attribute_value in event.items():
                if selected_event_attributes is not None:
                    if attribute_name not in selected_event_attributes:
                        continue
                trace_attribute_tuple = (attribute_name, 'event')
                if trace_attribute_tuple not in attributes_and_observed_types:
                    attributes_and_observed_types[trace_attribute_tuple] = set(
                        [_get_data_type(attribute_value)])
                else:
                    attributes_and_observed_types[trace_attribute_tuple].add(
                        _get_data_type(attribute_value))

    # check if any attribute has more than one type, return that attribute to the user
    failed = []
    triplet_list = []
    for (attribute_name, attribute_level), attribute_types in attributes_and_observed_types.items():
        if len(attribute_types) > 1:
            failed.append((attribute_name, attribute_level))
        triplet_list.append(
            (attribute_name, attribute_level, next(iter(attribute_types))))

    if len(failed) > 0:
        print(f'Detected multiple attribute types for the following attributes:')
        failed_types = itemgetter(failed)(attributes_and_observed_types)
        print(failed_types)

    return triplet_list


def get_simple_change_point_list_from_dictonary(change_point_explanations):
    """Simplifies the drift explanation dictionary into a list of tuples.

    Args:
        change_point_explanations: Dictionary with change point explanations.

    Returns:
        List with change point and change detector tuples.
    """
    # flatten explanations into single list
    change_point_explanations_list = sum(
        change_point_explanations.values(), [])

    # get cp tuples
    cp_tuple_list = [(cp_explanation['change_point'], cp_explanation['detector'])
                     for cp_explanation in change_point_explanations_list]

    return cp_tuple_list


def append_config_results(results_file_path, event_log_file_path, configuration_dict, results_dict, compute_time, experiment_name):
    """Append a result to the result file.

    Args:
        results_file_path: File path of the results file.
        event_log_file_path: File path to the event log.
        configuration_dict: Dictionary with all configurations.
        results_dict: Dictionary with all results.
        compute_time: Time it took for all computations
        experiment_name: Name of the experiment.
    """
    configuration_dict_prepended = {
        f'config_{key}': val for key, val in configuration_dict.items()}
    merged_dictionary = configuration_dict_prepended | results_dict

    merged_dictionary['config_event_log_file_path'] = event_log_file_path
    merged_dictionary['compute_time'] = compute_time
    merged_dictionary['experiment_name'] = experiment_name

    field_names = list(merged_dictionary.keys())

    # create results_file_path if not exists
    if not os.path.exists(os.path.dirname(results_file_path)):
        # Create a new directory because it does not exist
        os.makedirs(os.path.dirname(results_file_path))

    file_exists = os.path.isfile(results_file_path)

    with open(results_file_path, 'a') as f_object:
        dictwriter_object = csv.DictWriter(f_object, fieldnames=field_names)

        # create the header row if file did not exist
        if not file_exists:
            dictwriter_object.writeheader()

        # Pass the dictionary as an argument to the Writerow()
        dictwriter_object.writerow(merged_dictionary)

        # Close the file object
        f_object.close()


def get_all_files_in_dir(dir, include_files_in_subdirs=True):
    """Gets all files that are in a folder.

    Args:
        dir: Folder path.
        include_files_in_subdirs: Include files in subfolders as well?

    Return:
        List of all file paths.
    """
    # get all files in directory
    files_in_dir = os.listdir(dir)
    all_files = list()
    # Iterate over all the entries
    for entry in files_in_dir:
        # Create full path
        file_path = os.path.join(dir, entry)
        # If entry is a directory then get the list of files in this directory
        if include_files_in_subdirs and os.path.isdir(file_path):
            all_files = all_files + \
                get_all_files_in_dir(file_path, include_files_in_subdirs)
        else:
            all_files.append(file_path)

    return all_files


def get_examples_of_event_attributes(event_log, number_examples, for_event_attributes=None):
    """Get examples of values for event attributes.

    Args:
        event_log: PM4Py event log.
        number_examples: Number of example values to return.
        for_event_attributes: Optional list of event attributes to return examples for.

    Returns:
        Dataframe with example values.
    """
    # sample some traces
    sample_trace_numbers = np.random.choice(
        range(len(event_log)), number_examples, replace=False)
    sample_traces = [event_log[trace] for trace in sample_trace_numbers]

    # sample one event for each trace to add to the set of example event attributes
    event_attribute_values_list = []

    for trace in sample_traces:
        # choose a sample event from the trace
        event_number = np.random.choice(range(len(trace)))
        event = trace[event_number]

        # if event_attributes is set, only get the defined event attributes from the event
        filtered_event = {key: value for key,
                          value in event.items() if key in for_event_attributes}

        event_attribute_values_list.append(filtered_event)

    # place event_attribute_values_list into DataFrame
    event_attributes_example_df = pd.DataFrame(
    ).from_records(event_attribute_values_list)

    # return dataframe
    return event_attributes_example_df


def get_examples_of_trace_attributes(event_log, number_examples, for_trace_attributes=None):
        """Get examples of values for trace attributes.

    Args:
        event_log: PM4Py event log.
        number_examples: Number of example values to return.
        for_trace_attributes: Optional list of trace attributes to return examples for.

    Returns:
        Dataframe with example values.
    """
    # sample some traces
    sample_trace_numbers = np.random.choice(
        range(len(event_log)), number_examples, replace=False)
    sample_traces = [event_log[trace] for trace in sample_trace_numbers]

    # sample one event for each trace to add to the set of example event attributes
    trace_attribute_values_list = []
    for trace in sample_traces:
        # get the trace attributes
        trace_attributes = trace.attributes

        if for_trace_attributes is not None:
            # filter trace attributes for those in trace_attributes
            trace_attributes = {key: value for key, value in trace_attributes.items(
            ) if key in for_trace_attributes}

        trace_attribute_values_list.append(trace_attributes)

    # place event_attribute_values_list into DataFrame
    trace_attributes_example_df = pd.DataFrame(trace_attribute_values_list)

    # return dataframe
    return trace_attributes_example_df


def get_configurations(window_generator_types=['fixed', 'adaptive'],
                       window_sizes=[100, 150, 200],
                       thresholds=[0.05],
                       max_distances=[300],
                       slide_bys=[10, 20],
                       proportional_phis=[0.25, 0.5, 1],
                       proportional_rhos=[0.1]):
    """Create drift detection configurations from lists of input parameters.

    The configurations can be used in a parameter grid search.
    """
    # build all possible configuration:
    configurations = []
    for window_generator_type in window_generator_types:
        for window_size in window_sizes:
            for threshold in thresholds:
                for max_distance in max_distances:
                    for slide_by in slide_bys:
                        for proportional_phi in proportional_phis:
                            for proportional_rho in proportional_rhos:
                                configurations.append({
                                    'window_generator_type': window_generator_type,
                                    'window_size': window_size,
                                    'threshold': threshold,
                                    'max_distance': max_distance,
                                    'slide_by': slide_by,
                                    'proportional_phi': proportional_phi,
                                    'proportional_rho': proportional_rho
                                })
    return configurations


def perform_synthetic_experiments(experiment_name, configurations, input_path, results_path, delete_if_results_exist=False, limit_iterations=None):
    """Helper method for performing the experiments on the synthetic dataset.

    Args:
        experiment_name: Name of the experiment.
        configurations: List of configuration dictionaries.
        input_path: File path to input event log.
        results_path: Path to results CSV.
        delete_if_results_exist: Whether or not to delete the CSV file before writing to it.
        limit_iterations: Optional limit of experiments that are performed. Used for testing. 
    """

    # get the true change points and true change point explanations
    true_change_points = get_change_points_maardji_et_al_2013(10000)
    number_relevant_attributes = 5
    true_change_point_explanations = [
        (true_change_points[i], f'trace: relevant_attribute_{i+1:02d}') for i in range(number_relevant_attributes)]

    # load all event logs from the input path
    event_log_file_paths = get_all_files_in_dir(
        input_path, include_files_in_subdirs=True)

    # primary drift detector stays always the same
    primary_process_drift_detector = drift_detection.TrueKnownDD(
        true_change_points)

    # delete results file if exists
    if delete_if_results_exist:
        if os.path.exists(results_path):
            os.remove(results_path)

    # iterate all datasets with all settings
    for i, event_log_file_path in enumerate(event_log_file_paths):
        if limit_iterations is not None:
            if i >= limit_iterations:
                break

        print(f'Event log {i}')
        event_log = xes_importer.apply(event_log_file_path)

        for configuration in configurations:
            print(f'\nEvaluating configuration {configuration}')

            start_time = time.time()

            window_generator_type = configuration['window_generator_type']
            window_size = configuration['window_size']
            threshold = configuration['threshold']
            max_distance = configuration['max_distance']
            slide_by = configuration['slide_by']
            proportional_phi = configuration['proportional_phi']
            proportional_rho = configuration['proportional_rho']

            window_generator = None
            # build the secondary cdrift detector
            if window_generator_type == 'fixed':
                window_generator = windowing.FixedWG(
                    window_size, slide_by=slide_by)
            elif window_generator_type == 'adaptive':
                window_generator = windowing.AdaptiveWG(
                    window_size, slide_by=slide_by)

            phi = math.ceil(proportional_phi * window_size / slide_by)
            rho = math.ceil(proportional_rho * window_size / slide_by)

            change_point_extractor = change_point_extraction.PhiFilterCPE(
                threshold, phi, rho)

            attributes_and_types = get_attributes_and_types_for_synthetic_data()

            secondary_drift_detectors = drift_detection.get_attribute_drift_detectors(
                attributes_and_types,
                window_generator,
                change_point_extractor,
            )

            drift_explainer = drift_explanation.DriftExplainer(
                primary_process_drift_detector, secondary_drift_detectors)

            # calculate the drift explanations
            drift_explanation_result = drift_explainer.get_possible_drift_explanations(
                event_log, max_distance)

            # evaluate the change point explanations
            observed_drift_point_explanations_simple = get_simple_change_point_list_from_dictonary(
                drift_explanation_result.possible_drift_explanations)

            result = evaluation.evaluate_explanations(
                true_change_point_explanations, observed_drift_point_explanations_simple, max_distance=window_size)

            # get end time
            end_time = time.time()
            # get the compute time and write into results
            compute_time = end_time - start_time

            # write the configuration results to file
            append_config_results(results_path, event_log_file_path,
                                  configuration, result, compute_time, experiment_name)


def get_attribute_values_around_cp(event_log, attribute_level, attribute_name, change_point, window_width=200):
    """Retrieve attribute values before and after a change point.
    
    Args:
        event_log: A PM4Py event log.
        attribute_level: 'trace' or 'event'; Level of the attribute
        attribute_name: Name of the attribute.
        change_point: Change point to get windows before/after.
        window_width: Width of each window.
    """
    window_a_start = change_point-window_width+1
    window_b_start = change_point + 1
    # Analysis for feature 'Includes_subCases'
    # compare two windows with this feature
    print(f'Analysis of {attribute_level} attribute {attribute_name}')
    window_generator = windowing.FixedWG(
        window_width, window_offset=window_b_start-window_a_start, slide_by=1)
    window_a, window_b = next(window_generator.get_windows(
        event_log, start=window_a_start))

    feature_extractor = feature_extraction.AttributeFE(
        attribute_level=attribute_level, attribute_name=attribute_name)

    features_window_a = feature_extractor.extract(window_a.log)
    features_window_b = feature_extractor.extract(window_b.log)

    return (features_window_a, features_window_b)
