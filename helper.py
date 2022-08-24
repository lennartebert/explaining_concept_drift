# imports
import pm4py
import json
import os
import pandas as pd
import numpy as np
import csv

import numbers
from operator import itemgetter
from pm4py.util import xes_constants as xes
import math

import time

from opyenxes.data_in import XesXmlParser
from opyenxes.data_out import XesXmlSerializer

from processdrift import generate_attributes

from pm4py.objects.log.importer.xes import importer as xes_importer
from processdrift.framework import drift_detection
from processdrift.framework import drift_explanation
from processdrift.framework import feature_extraction
from processdrift.framework import population_comparison
from processdrift.framework import windowing
from processdrift.framework import evaluation
from processdrift.framework import change_point_extraction

def get_log(dataset_name):
    """Get the event log for a specified dataset.
    
    Available datasets:
    - 'bpi_challenge_2013_incidents'
    
    params:
        dataset_name: Name of dataset to import. See description for options.
    """
    
    if dataset_name == 'bpi_challenge_2013_incidents':
        path = 'data/real/bpi_challenge_2013_incidents.xes'
        log = pm4py.objects.log.importer.xes.importer.apply(path)
        return log
    
    return None

def opyenxes_read_xes(data_file_path, multiple_logs=False):
    """Reads an XES event log with opyenxes.
    
    Args:
        data_file_path: Path to data file.
        multiple_logs: Set to true if the XES file contains multiple logs.
    
    Return:
        opyenxes event log(s)
    """
    opyenxes_xes_parser = XesXmlParser.XesXmlParser()
    
    with open(data_file_path) as data_file:
        opyenxes_logs = opyenxes_xes_parser.parse(data_file)
    
    if multiple_logs:
        return opyenxes_logs
    else:
        return opyenxes_logs[0]
    
def opyenxes_write_xes(log, data_file_path):
    """Writes an XES event log with opyenxes.
    
    Args:
        data_file_path: Path to data file.
         
    """
    opyenxes_xes_serializer = XesXmlSerializer.XesXmlSerializer()
    
    # create path if doesn't exist
    if not os.path.exists(os.path.dirname(data_file_path)):
        # Create a new directory because it does not exist 
        os.makedirs(os.path.dirname(data_file_path))
    
    opyenxes_logs = None
    with open(data_file_path, 'w') as data_file:
        opyenxes_xes_serializer.serialize(log, data_file)

def update_data_dictionary(data_dictionary, data_dict_file_path='data/data_dict.json'):
    """Extend exisitng data dictionary with given data dictionary. 
    Existing values will be overwritten if the key matches.
    
    Args:
        data_dictionary: Dictionary of datafiles.
    """
    existing_data_dictionary = {}
    # read existing data dictionary
    if os.path.exists(data_dict_file_path):
        with open(data_dict_file_path) as existing_data_dict_file:
            existing_data_dictionary = json.load(existing_data_dict_file)

    # merge existing data dictionary and new data dicitionary. Overwrite existing data dictionray if key is same
    merged_data_dictionary = existing_data_dictionary | data_dictionary

    json_dump = json.dumps(merged_data_dictionary, indent='   ')

    with open(data_dict_file_path, 'w') as data_dict_file:
        data_dict_file.write(json_dump)

def get_data_dictionary(data_dict_file_path='data/data_dict.json'):
    """Get the data dictionary for the project
    
    Args:
        data_dict_path: Path to data dictionary.
    
    Returns:
        Data dictionary as Python dict.
    """
    # read data dictionary
    data_dictionary = json.load(open(data_dict_file_path))
    return data_dictionary

def get_data_information(data_file_path, data_dict_file_path='data/data_dict.json'):
    """Get the data information for a given data file path.
    
    Args:
        data_file_path: Path to data file.
        data_dict_path: Path to data dictionary.
    
    Returns:
        Data information as Python dict.
    """
    # read data dictionary
    data_dictionary = json.load(open(data_dict_file_path))
    
    # clean the data file path
    data_file_path = os.path.normpath(data_file_path)
    
    # get the data information
    data_information = data_dictionary[data_file_path]
    return data_information

def get_datasets_by_criteria(data_dict_file_path='data/data_dict.json', 
                                    file_path=None, 
                                    file_name=None, 
                                    drift_type=None, 
                                    dataset=None, 
                                    size=None, 
                                    is_synthetic=None, 
                                    has_generated_attributes=None,
                                    in_folder=None):
    """Get all datasets that match all of the given criteria."""
    data_dictionary = get_data_dictionary(data_dict_file_path=data_dict_file_path)
    
    # get a list of all not None conditions
    not_none_conditions = {}
    if file_path is not None: not_none_conditions['file_path'] = file_path
    if file_name is not None: not_none_conditions['file_name'] = file_name
    if drift_type is not None: not_none_conditions['drift_type'] = drift_type
    if dataset is not None: not_none_conditions['dataset'] = dataset
    if size is not None: not_none_conditions['size'] = size
    if is_synthetic is not None: not_none_conditions['is_synthetic'] = is_synthetic
    if has_generated_attributes is not None: not_none_conditions['has_generated_attributes'] = has_generated_attributes
    
    # filter the existing data dictionary
    filtered_dictionary = {}
    for index, data_info in data_dictionary.items():
        evaluations = [data_info[condition_field] == condition_value for condition_field, condition_value in not_none_conditions.items()]
        if in_folder is not None:
            normpath = os.path.normpath(in_folder)
            if os.path.normpath(data_info['file_path']).startswith(normpath):
                evaluations.append(True)
            else:
                evaluations.append(False)
        if all(evaluations):
            filtered_dictionary[index] = data_info
    return filtered_dictionary

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
    change_points = [int((i + 1)*size/10) for i in range(9)]
    return change_points

def add_synthetic_attributes(input_file_path,
    output_file_path,
    change_points,
    count_relevant_attributes,
    count_irrelevant_attributes,
    number_attribute_values=3,
    type_of_drift='sudden',
    type_of_change='mixed',
    standard_deviation_offset_explain_change_point=0):
    """Given an pm4py event log, add synthetic attribute data.
    """

    # load the log
    opyenxes_log = opyenxes_read_xes(input_file_path)
    
    # add synthetic attributes to log
    ag = generate_attributes.create_and_populate_attribute_generator(opyenxes_log, 
                                                                    change_points,
                                                                    count_relevant_attributes,
                                                                    count_irrelevant_attributes,
                                                                    number_attribute_values,
                                                                    type_of_drift,
                                                                    type_of_change,
                                                                    standard_deviation_offset_explain_change_point=standard_deviation_offset_explain_change_point)
    
    # save the log
    opyenxes_write_xes(ag.opyenxes_log, output_file_path)

    # return the change points
    return ag.change_point_explanations


def get_simple_change_point_format_from_data_info(data_info):
    complex_cp_explanations = data_info['change_point_explanations']
    simple_cp_explanations = []
    for complex_cp_explanation in complex_cp_explanations:
        cp_location = complex_cp_explanation['change_point']
        attribute_change = complex_cp_explanation['attribute_name']
        simple_cp_explanations.append((cp_location, attribute_change))
    return simple_cp_explanations


def get_attributes_and_types_for_synthetic_data(relevant_attributes=5, irrelevant_attributes=5):
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
    

def get_data_type(attribute_value):
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
        trace_attributes: Only get date types for the specified trace attributes. If None, all trace attributes are returned.
        event_attributes: Only get date types for the specified event attributes. If None, all event attributes are returned.

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
                attributes_and_observed_types[trace_attribute_tuple] = set([get_data_type(attribute_value)])
            else:
                attributes_and_observed_types[trace_attribute_tuple].add(get_data_type(attribute_value))
        
        # get the event attribute types
        for event in trace:
            for attribute_name, attribute_value in event.items():
                if selected_event_attributes is not None:
                    if attribute_name not in selected_event_attributes:
                        continue
                trace_attribute_tuple = (attribute_name, 'event')
                if trace_attribute_tuple not in attributes_and_observed_types:
                    attributes_and_observed_types[trace_attribute_tuple] = set([get_data_type(attribute_value)])
                else:
                    attributes_and_observed_types[trace_attribute_tuple].add(get_data_type(attribute_value))
    
    # remove standard trace and event attributes
    # attributes_and_observed_types.pop((xes.DEFAULT_TRACEID_KEY, 'trace'), None)
    # attributes_and_observed_types.pop((xes.DEFAULT_START_TIMESTAMP_KEY, 'trace'), None)
    # attributes_and_observed_types.pop((xes.DEFAULT_TRANSITION_KEY, 'event'), None)
    # attributes_and_observed_types.pop((xes.DEFAULT_TIMESTAMP_KEY, 'event'), None)
    # attributes_and_observed_types.pop((xes.DEFAULT_NAME_KEY, 'event'), None)

    # check if any attribute has more than one type, return that attribute to the user
    failed = []
    triplet_list = []
    for (attribute_name, attribute_level), attribute_types  in attributes_and_observed_types.items():
        if len(attribute_types) > 1:
            failed.append((attribute_name, attribute_level))
        triplet_list.append((attribute_name, attribute_level, next(iter(attribute_types))))
    
    if len(failed) > 0:
        print(f'Detected multiple attribute types for the following attributes:')
        failed_types = itemgetter(failed)(attributes_and_observed_types)
        print(failed_types)
    
    return triplet_list


def get_simple_change_point_list_from_dictonary(change_point_explanations):
    # flatten explanations into single list
    change_point_explanations_list = sum(change_point_explanations.values(), [])

    # get cp tuples
    cp_tuple_list = [(cp_explanation['change_point'], cp_explanation['detector']) for cp_explanation in change_point_explanations_list]
    
    return cp_tuple_list

def append_config_results(results_file_path, event_log_file_path, configuration_dict, results_dict, compute_time, experiment_name):
    configuration_dict_prepended = {f'config_{key}': val for key, val in configuration_dict.items()}
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
    # get all files in directory
    files_in_dir = os.listdir(dir)
    all_files = list()
    # Iterate over all the entries
    for entry in files_in_dir:
        # Create full path
        file_path = os.path.join(dir, entry)
        # If entry is a directory then get the list of files in this directory 
        if include_files_in_subdirs and os.path.isdir(file_path):
            all_files = all_files + get_all_files_in_dir(file_path, include_files_in_subdirs)
        else:
            all_files.append(file_path)
                
    return all_files

def get_examples_of_event_attributes(event_log, number_examples, for_event_attributes=None):
    """Get examples for all event attributes.
    """
    # sample some traces
    sample_trace_numbers = np.random.choice(range(len(event_log)), number_examples, replace=False)
    sample_traces = [event_log[trace] for trace in sample_trace_numbers]

    # sample one event for each trace to add to the set of example event attributes
    event_attribute_values_list = []
    
    for trace in sample_traces:
        # choose a sample event from the trace
        event_number = np.random.choice(range(len(trace)))
        event = trace[event_number]

        # if event_attributes is set, only get the defined event attributes from the event
        filtered_event = {key:value for key, value in event.items() if key in for_event_attributes}

        event_attribute_values_list.append(filtered_event)

    # place event_attribute_values_list into DataFrame
    event_attributes_example_df = pd.DataFrame().from_records(event_attribute_values_list)

    # return dataframe
    return event_attributes_example_df


def get_examples_of_trace_attributes(event_log, number_examples, for_trace_attributes=None):
    # sample some traces
    sample_trace_numbers = np.random.choice(range(len(event_log)), number_examples, replace=False)
    sample_traces = [event_log[trace] for trace in sample_trace_numbers]

    # sample one event for each trace to add to the set of example event attributes
    trace_attribute_values_list = []
    for trace in sample_traces:
        # get the trace attributes
        trace_attributes = trace.attributes

        if for_trace_attributes is not None:
            # filter trace attributes for those in trace_attributes
            trace_attributes  = {key: value for key, value in trace_attributes.items() if key in for_trace_attributes}
        
        trace_attribute_values_list.append(trace_attributes)
    
    # place event_attribute_values_list into DataFrame
    trace_attributes_example_df = pd.DataFrame(trace_attribute_values_list)

    # return dataframe
    return trace_attributes_example_df

def get_configurations(window_generator_types=['fixed', 'adaptive'],
                        window_sizes=[100, 150, 200],
                        thresholds = [0.05],
                        max_distances = [300],
                        slide_bys = [10, 20],
                        proportional_phis = [0.25, 0.5, 1],
                        proportional_rhos = [0.1]):
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
    # get the true change points and true change point explanations
    true_change_points = get_change_points_maardji_et_al_2013(10000)
    number_relevant_attributes = 5
    true_change_point_explanations = [(true_change_points[i], f'trace: relevant_attribute_{i+1:02d}') for i in range(number_relevant_attributes)]
    
    # load all event logs from the input path
    event_log_file_paths = get_all_files_in_dir(input_path, include_files_in_subdirs=True)

    # primary drift detector stays always the same
    primary_process_drift_detector = drift_detection.TrueKnownDD(true_change_points)

    # delete results file if exists
    if delete_if_results_exist:
        if os.path.exists(results_path):
            os.remove(results_path)
    
    # iterate all datasets with all settings
    for i, event_log_file_path in enumerate(event_log_file_paths):
        if limit_iterations is not None:
            if i >= limit_iterations: break
        
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
            # build the secondary drift detector
            if window_generator_type == 'fixed':
                window_generator = windowing.FixedWG(window_size, slide_by=slide_by)
            elif window_generator_type == 'adaptive':
                window_generator = windowing.AdaptiveWG(window_size, slide_by=slide_by)
            
            phi = math.ceil(proportional_phi * window_size / slide_by)
            rho = math.ceil(proportional_rho * window_size / slide_by)
            
            change_point_extractor = change_point_extraction.PhiFilterCPE(threshold, phi, rho)

            attributes_and_types = get_attributes_and_types_for_synthetic_data()

            secondary_drift_detectors = drift_detection.get_attribute_drift_detectors(
                                                                                attributes_and_types,
                                                                                window_generator,
                                                                                change_point_extractor,
                                                                                )
            
            drift_explainer = drift_explanation.DriftExplainer(primary_process_drift_detector, secondary_drift_detectors)

            # calculate the drift explanations
            drift_explanation_result = drift_explainer.get_possible_drift_explanations(event_log, max_distance)
            
            # evaluate the change point explanations
            observed_drift_point_explanations_simple =  get_simple_change_point_list_from_dictonary(drift_explanation_result.possible_drift_explanations)    

            result = evaluation.evaluate_explanations(true_change_point_explanations, observed_drift_point_explanations_simple, max_distance=window_size)
            
            # get end time
            end_time = time.time()
            # get the compute time and write into results
            compute_time = end_time - start_time
            
            # write the configuration results to file
            append_config_results(results_path, event_log_file_path, configuration, result, compute_time, experiment_name)


def get_attribute_values_around_cp(event_log, attribute_level, attribute_name, change_point, window_width = 200):
    window_a_start = change_point-window_width+1
    window_b_start = change_point + 1
    # Analysis for feature 'Includes_subCases'
    # compare two windows with this feature
    print(f'Analysis of {attribute_level} attribute {attribute_name}')
    window_generator = windowing.FixedWG(window_width, window_offset=window_b_start-window_a_start, slide_by=1)
    window_a, window_b = next(window_generator.get_windows(event_log, start=window_a_start))
    
    feature_extractor = feature_extraction.AttributeFE(attribute_level=attribute_level, attribute_name=attribute_name)

    features_window_a = feature_extractor.extract(window_a.log)
    features_window_b = feature_extractor.extract(window_b.log)
    
    return (features_window_a, features_window_b)


# def save_experiment_results(experiment_name,
#     dataset,
#     ):
#     result = {}
#     result['dataset_info'] = dataset_info
#     result['attribute_drift_detector'] = attribute_drift_detector
#     result['aggregated_results'] = aggregated_results
#     return

# def write_results_complex(dataset_name,
#     dataset_size,
#     configuration,
#     window_strategy,
#     window_size,
#     window_slide_by
#     pop_comparer,
#     comparison_threshold,
#     max_distance,
#     number_experiments,
#     precision,
#     recall,
#     f1_score,
#     mean_lag,
#     results_file_path
#     ):

#     result = {}
#     result['dataset_name'] = dataset_name
#     result['dataset_size'] = dataset_size
#     result['configuration'] = configuration
#     result['window_strategy'] = window_strategy
#     result['window_size'] = window_size
#     result['window_slide_by'] = window_slide_by
#     result['pop_comparer'] = pop_comparer
#     result['comparison_threshold'] = comparison_threshold
#     result['max_distance'] = max_distance
#     result['number_experiments'] = number_experiments
#     result['precision'] = precision
#     result['recall'] = recall
#     result['f1_score'] = f1_score
#     result['mean_lag'] = mean_lag
    
    
#     if os.path.exists(results_file_path):
#         with open(results_file_path,'r+') as file:
#             # First we load existing data into a dict.
#             all_results_data = json.load(file)
#             # Join new_data with file_data inside emp_details
#             all_results_data.append(result)
#             # Sets file's current position at offset.
#             file.seek(0)
#             # convert back to json.
#             json.dump(all_results_data, file, indent = 4)
#     else:
#         all_results_data = [result]

#         results_path = os.path.dirname(results_file_path)
#         # Check whether the specified path exists or not
#         isExist = os.path.exists(results_path)

#         if not isExist:
#             # Create a new directory because it does not exist 
#             os.makedirs(results_path)

#         # create new file
#         with open(results_file_path, 'w') as file:
#             json.dump(all_results_data, file, indent = 4)
