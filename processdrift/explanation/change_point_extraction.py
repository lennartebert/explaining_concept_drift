"""Module for the detecting change points from a time series of comparison measures.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ChangePointExtractor(ABC):
    """Implements a strategy for extracting change points from a change series."""
    @abstractmethod
    def get_change_points(self, change_series):
        """Get the change points from a change series.

        Args:
            change_series: Pandas series of similarity measure (e.g., p-values).

        Returns:
            List of change points
        """
        pass


class PhiFilterCPE(ChangePointExtractor):
    """Applies a phi-filter to extract change points.

    The general idea is that there need to be a minimum of "phi" 
    observations below a threshold to count a change.
    """

    def __init__(self, threshold, phi, rho=0):
        """Initialize the phi filer.

        Args:
            threshold: Lower threshold for the similarity measure. E.g., 0.05 for p-values.
            phi: Minimum observations below the threshold.
            rho: In a streak of observations below the threshold, 
                number of observations that can be below the threshold.
        """
        self.threshold = threshold
        self.phi = phi
        self.rho = rho

    def get_change_points(self, change_series):
        """Get the change points from a change series.

        Args:
            change_series: Pandas series of similarity measure (e.g., p-values).

        Returns:
            List of change points
        """
        # for each row, get whether its value is of threshold or lower
        below_threshold_series = change_series <= self.threshold

        # the index of the change series does not need to be continuous and could be sth. like
        # like the case count. Therefore, reset the index and save the original one.
        original_index = change_series.index

        # reset the index
        below_threshold_series = below_threshold_series.reset_index(drop=True)

        change_points = set()

        phi_filter_count = 0
        rho_filter_count = 0
        streak_beginning = None
        streak = 0
        for i, is_below_threshold in below_threshold_series.iteritems():
            if is_below_threshold:
                # save the index of the beginning of the streak
                if streak_beginning == None:
                    streak_beginning = i

                streak += 1
                phi_filter_count += 1
                rho_filter_count = 0
            else:
                rho_filter_count += 1
                phi_filter_count = 0
                if rho_filter_count > self.rho:
                    streak_beginning = None
                    streak = 0

            if streak == self.phi:
                # get the original index and append it to the change points
                original_index_start_streak = original_index[streak_beginning]

                change_points.add(original_index_start_streak)

        return list(change_points)
