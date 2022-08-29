"""Helper module for event log feature generation.

If features are zero-dimensional, functions return a pandas Series with one row.
If features are one-dimensional, functions return a pandas Series.
If features are two-dimensional, functions return a pandas Dataframe.
"""

import copy

import networkx as nx
import numpy as np
import pandas as pd


def _get_traces(log, activity_name_field='concept:name'):
    """Get traces from a log as list of activity names.

    Args:
        log: A pm4py Eventlog
        activity_name_field: Field name in the event log to identify the activity name.
    """
    traces = []
    for line in log:
        trace = []
        for event in line:
            trace.append(event[activity_name_field])
        traces.append(trace)
    return traces


def get_activity_occurence_counts(log, activity_name_field='concept:name'):
    """Count the number of occurences per activity.

    Args:
        log: A pm4py Eventlog.
        activity_name_field: Field name in the event log to identify the activity name.

    Returns:
        Number of occurences for each activity as Pandas Series.
    """
    traces = _get_traces(log, activity_name_field)

    return _get_activity_occurence_counts(traces)


def _get_activity_occurence_counts(traces):
    """Count the number of occurences per activity.

    Args:
        traces: List of traces to calculate causality count from.

    Returns:
        Number of occurences for each activity as Pandas Series.
    """
    activity_occurences = {}
    for trace in traces:
        for activity in trace:
            if activity not in activity_occurences:
                activity_occurences[activity] = 1
            else:
                activity_occurences[activity] += 1

    return pd.Series(activity_occurences)


def get_causality_counts(log, direction='followed_by', activity_name_field='concept:name'):
    """ For a log, gets the causality counts in the specified direction.

    Args:
        log: A pm4py Eventlog.
        direction: "followed_by" or "preceded_by". Direction of causality.
        activity_name_field: Field name in the event log to identify the activity name.

    Returns:
        Dataframe with causality counts for all activities.
    """
    traces = _get_traces(log, activity_name_field)
    return _get_causality_counts(traces, direction=direction)


def _get_causality_counts(traces, direction='followed_by'):
    """ For a set of traces, gets the causality counts in the specified direction.

    Args:
        traces: List of traces to calculate causality count from.
        direction: "followed_by" or "preceded_by". Direction of causality.

    Returns:
        Dataframe with causality counts for all activities.
    """

    all_activities = set()  # set of all activities
    causality_dict = {}
    for trace in traces:
        trace_length = len(trace)
        for i in range(trace_length - 1):
            all_activities.add(trace[i])
            # do not interate for the last item in a trace (no following/precedes relationship)
            if i == trace_length - 1:
                continue
            for j in range(trace_length - i - 1):
                activity_1_pos = None
                activity_2_pos = None

                # to be read activity_1 followed_by/preceded_by activity_2
                if direction == 'followed_by':
                    activity_1_pos = i
                    activity_2_pos = i + j + 1
                elif direction == 'preceded_by':
                    activity_1_pos = trace_length - i - 1
                    activity_2_pos = trace_length - i - j - 2

                activity_1 = trace[activity_1_pos]
                activity_2 = trace[activity_2_pos]

                # add activity 1 to causality dict if doesn't exist
                if activity_1 not in causality_dict:
                    causality_dict[activity_1] = {}

                # count toward the relationship count
                if activity_2 not in causality_dict[activity_1]:
                    causality_dict[activity_1][activity_2] = 1
                else:
                    causality_dict[activity_1][activity_2] += 1

    # build the causality count dataframe
    causality_df = pd.DataFrame().from_dict(causality_dict, orient='index')

    # replace nan-values with 0
    causality_df = causality_df.fillna(0)

    # make sure all activities have a row and a column
    for activity in all_activities:
        if activity not in causality_df.columns:
            causality_df[activity] = 0
        if activity not in list(causality_df.index.values):
            causality_df.loc[activity] = 0

    # convert all values to integers
    causality_df = causality_df.apply(pd.to_numeric, downcast='integer')

    return causality_df


def get_sna_relationships(log, direction='followed_by', activity_name_field='concept:name'):
    """Gets the sometimes always never relationships for a causality count dictionary.

    Args:
        log: A pm4py Eventlog.
        direction: "followed_by" or "preceded_by". Direction of causality.
        activity_name_field: Field name in the event log to identify the activity name.

    Returns:
        Dataframe with relationship description (sometimes (s), always (a), never (n)).
    """
    traces = _get_traces(log, activity_name_field)
    causality_counts = _get_causality_counts(traces, direction)
    activity_occurence_counts = _get_activity_occurence_counts(traces)
    return _get_sna_relationships(causality_counts, activity_occurence_counts)


def _get_sna_relationships(causality_counts, activity_occurence_counts):
    """Gets the sometimes always never relationships for a causality count dictionary.

    Args:
        causality_counts: Dataframe of causality counts.
        activity_occurence_counts: Series of activity occurence counts.

    Returns:
        Dataframe with relationship description (sometimes (s), always (a), never (n)).
    """
    sna_dict = {}
    for activity_1, activity_2_counts in causality_counts.iterrows():
        # add activity 1 to sna dictionary
        sna_dict[activity_1] = {}

        for activity_2, activity_2_count in activity_2_counts.items():
            relationship = None
            if activity_2_count == activity_occurence_counts[activity_2]:
                relationship = 'a'
            elif activity_2_count > 0:
                relationship = 's'
            else:
                relationship = 'n'

            sna_dict[activity_1][activity_2] = relationship

    # convert to pandas Dataframe
    sna_df = pd.DataFrame().from_dict(sna_dict, orient='index')
    return sna_df


def get_relationship_type_counts(log, direction='followed_by', activity_name_field='concept:name'):
    """Gets the relation type counts per activity.

    Args:
        log: A pm4py Eventlog.
        direction: "followed_by" or "preceded_by". Direction of causality.
        activity_name_field: Field name in the event log to identify the activity name.

    Returns:
        Dataframe with the count of each relationship type per activity.
    """
    traces = _get_traces(log, activity_name_field)
    causality_counts = _get_causality_counts(traces, direction)
    activity_occurence_counts = _get_activity_occurence_counts(traces)
    sna_relationships = _get_sna_relationships(
        causality_counts, activity_occurence_counts)
    return _get_relation_type_counts(sna_relationships)


def get_bi_directional_relation_type_counts(log, activity_name_field='concept:name'):
    """Gets the relation type counts per activity.

    Args:
        log: A pm4py Eventlog.
        direction: "followed_by" or "preceded_by". Direction of causality.
        activity_name_field: Field name in the event log to identify the activity name.

    Returns:
        Dataframe with the count of each relationship type per activity.
    """
    traces = _get_traces(log, activity_name_field)

    # get the type relationship type counts for the followed by relationships
    causality_counts_followed_by = _get_causality_counts(
        traces, direction='followed_by')
    activity_occurence_counts_followed_by = _get_activity_occurence_counts(
        traces)
    sna_relationships_followed_by = _get_sna_relationships(
        causality_counts_followed_by, activity_occurence_counts_followed_by)
    relationship_type_counts_followed_by = _get_relation_type_counts(
        sna_relationships_followed_by)

    # get the type relationship type counts for the preceded by relationships
    causality_counts_preceded_by = _get_causality_counts(
        traces, direction='preceded_by')
    activity_occurence_counts_preceded_by = _get_activity_occurence_counts(
        traces)
    sna_relationships_preceded_by = _get_sna_relationships(
        causality_counts_preceded_by, activity_occurence_counts_preceded_by)
    relationship_type_counts_preceded_by = _get_relation_type_counts(
        sna_relationships_preceded_by)

    # merge the followed/preceded by relationship type counts
    relationship_type_counts_followed_by = relationship_type_counts_followed_by.add_prefix(
        'followed_by_')
    relationship_type_counts_preceded_by = relationship_type_counts_preceded_by.add_prefix(
        'preceded_by_')

    # join both on the index
    relationship_type_counts = relationship_type_counts_followed_by.merge(
        relationship_type_counts_preceded_by, how='outer', left_index=True, right_index=True)
    return relationship_type_counts


def _get_relation_type_counts(sna_dataframe):
    """Gets the relation type counts per activity.

    Args:
        sna_dataframe: Dataframe indicating sometime/always/never relationship between two activities.

    Returns:
        Dataframe with the count of each relationship type per activity.
    """
    activity_rc = {}
    for i, row in sna_dataframe.iterrows():
        # count sometimes, always and never
        activity_rc[i] = row.value_counts().to_dict()

        if 'n' not in activity_rc[i]:
            activity_rc[i]['n'] = 0

        if 'a' not in activity_rc[i]:
            activity_rc[i]['a'] = 0

        if 's' not in activity_rc[i]:
            activity_rc[i]['s'] = 0

    return pd.DataFrame().from_dict(activity_rc, orient='index')


def get_relational_entropy(log, direction='followed_by', activity_name_field='concept:name'):
    """Get the relational entropy for a relation type count dataframe.

   Args:
        log: A pm4py Eventlog.
        direction: "followed_by" or "preceded_by". Direction of causality.
        activity_name_field: Field name in the event log to identify the activity name.

    Returns:
        Series with relation type entropy for each activity.
    """
    traces = _get_traces(log, activity_name_field)
    causality_counts = _get_causality_counts(traces, direction)
    activity_occurence_counts = _get_activity_occurence_counts(traces)
    sna_relationships = _get_sna_relationships(
        causality_counts, activity_occurence_counts)
    relationship_type_counts = _get_relation_type_counts(sna_relationships)
    return _get_relational_entropy(relationship_type_counts)


def _get_relational_entropy(relationship_type_counts_df):
    """Get the relational entropy for a relation type count dataframe.

    Args:
        relationship_type_counts_df: Dataframe with relation type counts.

    Returns:
        Dataframe with relation type entropy for each activity.
    """
    activity_count = len(relationship_type_counts_df.index)

    activity_relation_entropy = {}

    for activity, row in relationship_type_counts_df.iterrows():
        # get count for each relation type
        ca = row['a']
        cs = row['s']
        cn = row['n']

        # get probability for each relation type
        pa = ca / activity_count
        ps = cs / activity_count
        pn = cn / activity_count

        entropy = 0
        for p in [pa, ps, pn]:
            if p != 0:  # probabilities of 0 are ignored to avoid taking log(0)
                entropy += -p * np.log2(p)

        activity_relation_entropy[activity] = entropy

    relation_entropy_series = pd.Series(activity_relation_entropy)
    return(relation_entropy_series)


def get_alpha_direct_relationships(traces, direction='precedes'):
    """Get the alpha directly precedes/succeeds relationships in an event log as a list of tuples.

    In an event log with a trace ['a', 'b', 'c'], 'a' precedes 'b'. From the other direction, 'b' succeeds 'a'.

    Args:
        log: A pm4py event log.
        direction: 'precedes' or 'succeeds'. Direction of the alpha directly follows relationship.

    Returns:
        List of tuples with alpha directly follows/precedes relationships.
    """
    traces = copy.deepcopy(traces)
    direct_relationship = set()

    # iterate all traces
    for trace in traces:
        # revert sorting of trace if direction is 'succeeds'
        if direction == 'succeeds':
            trace.reverse()

        last_activity = None
        for activity in trace:
            # handle first iteration
            if last_activity is None:
                last_activity = activity
                continue
            direct_relationship.add((last_activity, activity))

            last_activity = activity

    return direct_relationship


def get_concurrency(traces):
    edges_precede = get_alpha_direct_relationships(
        traces, direction='precedes')
    edges_succeed = get_alpha_direct_relationships(
        traces, direction='succeeds')
    concurrency = edges_precede.intersection(edges_succeed)
    return concurrency


def get_concurrency_dict(concurrency_tuples):
    result = {}
    for i in concurrency_tuples:
        result.setdefault(i[0], []).append(i[1])
    return result


def get_runs(traces):
    global_concurrency = get_concurrency(traces)
    concurrency_dict = get_concurrency_dict(global_concurrency)

    # build a list of runs
    runs = []

    for trace in traces:
        edges = set()
        for index_activity_1 in range(len(trace) - 1):
            index_activity_2 = index_activity_1 + 1

            activity_1 = trace[index_activity_1]
            activity_2 = trace[index_activity_2]
            edges.add((activity_1, activity_2))

        run = set()
        for edge in edges:
            activity_1 = edge[0]
            activity_2 = edge[1]

            # do not add edge if activity_1 and activity_2 are the same (this would be removed in transitivity reduction anyways)
            if activity_1 == activity_2:
                continue

            if activity_1 in concurrency_dict:
                for conccurency in concurrency_dict[activity_1]:
                    if conccurency == activity_2:
                        run.add((activity_1, activity_2))
                        run.add((activity_2, activity_1))
                    else:
                        run.add(edge)
                        run.add((conccurency, activity_2))
            if activity_2 in concurrency_dict:
                for concurrency in concurrency_dict[activity_2]:
                    if concurrency == activity_1:
                        run.add((activity_1, activity_2))
                        run.add((activity_2, activity_1))
                    else:
                        run.add(edge)
                        run.add((activity_1, concurrency))
            else:
                run.add(edge)

        run_list = sorted(run)
        run_string = str(run_list)
        runs.append(run_string)

    return runs
