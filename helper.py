# imports
import pm4py

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