import numpy as np


class Observations:
    def __init__(self, times, filters, data_magnitudes, magnitude_errors, distance):
        """
        Static object to hold the observations for a kilonova
        :param times:
        :param filters:
        :param data_magnitudes:
        :param magnitude_errors:
        :param distance:
        """

        self.times = times
        self.filters = filters
        self.filters_unique = np.unique(filters)
        self.data_magnitudes = data_magnitudes
        self.magnitude_errors = magnitude_errors
        self.upper_limit = np.invert(np.isfinite(self.magnitude_errors))
        self.upper_limit_indices = np.where(self.upper_limit)[0]
        self.distance = distance
