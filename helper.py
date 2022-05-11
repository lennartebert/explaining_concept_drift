# imports
import pm4py
import json
import os

from opyenxes.data_in import XesXmlParser
from opyenxes.data_out import XesXmlSerializer

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

    with open(data_dict_file_path, 'w') as data_dict_file:
        json_dump = json.dumps(merged_data_dictionary, indent='   ')
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

def get_datasets_by_criteria(data_dict_file_path='data/data_dict.json', file_path=None, file_name=None, drift_type=None, dataset=None, size=None, is_synthetic=None, has_generated_attributes=None):
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