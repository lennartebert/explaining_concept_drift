"""This module serves to evaluate drift explainers and drift detectors.
"""

def evaluate_explanations(true_change_explanations, detected_change_explanations, max_distance=0):
    """Get the precision, recall, f1 and lags for lists of change explanations.

    The list of change explanations are of the following format:
        [(change_point_location, attribute_name), (..., ...), ...]

    Args:
        true_change_explanations: List of true change point explanation tuples. See method description for format.
        detected_change_explanations: List of detected change point explanation tuples. See method description for format.
        max_distance: max distance for true and detected change point to count as correct identification.

    Returns:
        Dictionary with evaluation for explanations.
    """
    number_of_true_changes = len(true_change_explanations)
    number_of_detections = len(detected_change_explanations)

    number_of_correct_detections = 0
    lags = []

    # sort true and detected changes
    true_change_explanations.sort()
    detected_change_explanations.sort()

    # simple iterative looping through the true and detected change points
    for true_change in true_change_explanations:
        true_change_point = true_change[0]
        true_change_attribute = true_change[1]
        for detected_change in detected_change_explanations:
            detected_change_point = detected_change[0]
            detected_change_attribute = detected_change[1]
            # check if detected change point is within max_distance to detected change point
            # and whether the correct attribute was identified
            if abs(true_change_point - detected_change_point) <= max_distance \
                and true_change_attribute == detected_change_attribute:
                number_of_correct_detections += 1
                lags.append(true_change_point - detected_change_point)

                # we can break here because the detected change points have been sorted
                break
    
    precision = None
    if number_of_detections > 0:
        precision = number_of_correct_detections / number_of_detections
    
    recall = None
    if number_of_true_changes > 0:
        recall = number_of_correct_detections / number_of_true_changes

    # get the f1 score (harmonic mean of recall and precision)
    # only calcualte f1 if precision and recall are both > 0
    f1_score = None
    if (recall is not None) and (precision is not None) and (recall > 0) and (precision > 0):
        f1_score =  2 / ((1/recall) + (1/precision))
    
    # only calculate the mean lag if there where any lags
    mean_lag = None
    if len(lags) >= 1:
        mean_lag = sum(lags) / len(lags)

    evaluation_result = {
        'number_of_correct_detections': number_of_correct_detections,
        'number_of_true_changes': number_of_true_changes,
        'number_of_detections': number_of_detections,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_lag': mean_lag,
        'all_lags': lags
    }

    return evaluation_result

# aggregate list of evaluation results TODO: docstring
def aggregate_cp_explanation_results(results_list):
    all_correct_detections = 0
    all_true_changes = 0
    all_detections = 0
    all_lags = []
    all_number_experiments = 0

    for result in results_list:
        all_correct_detections += result['number_of_correct_detections']
        all_true_changes += result['number_of_true_changes']
        all_detections += result['number_of_detections']
        all_lags = all_lags + result['all_lags']
        if 'number_experiments' in result:
            all_experiment_counts += result['number_experiments']

    all_number_experiments += len(results_list)

    precision = None
    if all_detections > 0:
        precision = all_correct_detections / all_detections
    
    recall = None
    if all_true_changes > 0:
        recall = all_correct_detections / all_true_changes

    # get the f1 score (harmonic mean of recall and precision)
    # only calcualte f1 if precision and recall are both > 0
    f1_score = None
    if (recall is not None) and (precision is not None) and (recall > 0) and (precision > 0):
        f1_score =  2 / ((1/recall) + (1/precision))
    
    # only calculate the mean lag if there where any lags
    mean_lag = None
    if len(all_lags) >= 1:
        mean_lag = sum(all_lags) / len(all_lags)

    evaluation_result = {
        'number_of_correct_detections': all_correct_detections,
        'number_of_true_changes': all_true_changes,
        'number_of_detections': all_detections,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_lag': mean_lag,
        'lags': all_lags,
        'number_experiments': all_number_experiments
    }
    
    return evaluation_result

def evaluate_single_detector(true_change_points, detected_change_points, max_distance=0):
    """Get the precision, recall, f1 and lags for list of true change points.

    The list of change explanations are of the following format:
        [change_point_location_1, change_point_location_2, ...]

    Args:
        true_change_points: List of true change point locations. See method description for format.
        detected_change_points: List of detected change point locations. See method description for format.
        max_distance: max distance for true and detected change point to count as correct identification.

    Returns:
        Dictionary with evaluation for explanations.
    """
    number_of_true_changes = len(true_change_points)
    number_of_detections = len(detected_change_points)

    number_of_correct_detections = 0
    lags = []

    # sort true and detected changes
    true_change_points.sort()
    detected_change_points.sort()

    # simple iterative looping through the true and detected change points
    for true_change_point in true_change_points:
        for detected_change_point in detected_change_points:
            # check if detected change point is within max_distance to detected change point
            if abs(true_change_point - detected_change_point) <= max_distance:
                number_of_correct_detections += 1
                lags.append(detected_change_point - true_change_point)

                # we can break here because the detected change points have been sorted
                break

    precision = number_of_correct_detections / number_of_detections
    recall = number_of_correct_detections / number_of_true_changes

    # get the f1 score (harmonic mean of recall and precision)
    f1_score =  2 / ((1/recall) + (1/precision))
    mean_lag = sum(lags) / len(lags)

    evaluation_result = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_lag': mean_lag,
        'all_lags': lags
    }

    return evaluation_result
