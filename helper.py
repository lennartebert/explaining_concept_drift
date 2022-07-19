# imports
import pm4py
import json
import os
import pandas as pd
import numpy as np
import csv

from opyenxes.data_in import XesXmlParser
from opyenxes.data_out import XesXmlSerializer

from processdrift import generate_attributes

def get_trace_attributes(log):
    """Get the trace level attributes from a log.
    
    TODO: implement get event attributes
    
    Args:
        log: A pm4py Eventlog
    
    Returns:
        A dictionary with the attribute name as key and series with trace attributes.
    """
    # get all trace attributes as one dataframe
    trace_attributes = {}
    for i, trace in enumerate(log):
        trace_attributes[i] = trace.attributes
    
    trace_attribute_df = pd.DataFrame().from_dict(trace_attributes, orient='index')
    
    # than convert the attributes back to a dictionary of series
    attributes_dict = {}
    for column_name in trace_attribute_df.columns:
        attribute_series = trace_attribute_df[column_name]
        
        # the type of each attribute should either be string or Numeric
        is_numeric = np.issubdtype(attribute_series.dtype, np.number)
        # if type is not numeric, convert to string
        if not is_numeric:
            attribute_series = attribute_series.astype('category')
            
        attributes_dict[column_name] = attribute_series
    
    return attributes_dict

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


def get_simple_change_point_list_from_dictonary(change_point_explanations):
    # flatten explanations into single list
    change_point_explanations_list = sum(change_point_explanations.values(), [])

    # get cp tuples
    cp_tuple_list = [(cp_explanation['change_point'], cp_explanation['detector']) for cp_explanation in change_point_explanations_list]
    
    return cp_tuple_list

def append_config_results(results_file_path, event_log_file_path, configuration_dict, results_dict, compute_time):
    configuration_dict_prepended = {f'config_{key}': val for key, val in configuration_dict.items()}
    merged_dictionary = configuration_dict_prepended | results_dict

    merged_dictionary['config_event_log_file_path'] = event_log_file_path
    merged_dictionary['compute_time'] = compute_time

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
