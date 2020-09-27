import numpy as np


class _Buffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer: np.ndarray = np.array([])

        self.buffer_max: float
        self.buffer_max_log: float
        self.buffer_min: float
        self.buffer_min_log: float
        self.buffer_max_no_outliers: float
        self.buffer_max_no_outliers_log: float
        self.buffer_min_no_outliers: float
        self.buffer_min_no_outliers_log: float

    def add_values(self, values):
        self.buffer = np.append(self.buffer, values=values)
        if len(self.buffer) > self.max_size:  # Pop oldest values
            difference = len(self.buffer) - self.max_size
            self.buffer = self.buffer[difference:]

        self.recalculate_statistics()

    def recalculate_statistics(self):
        self.buffer_max = np.amax(self.buffer)
        self.buffer_min = np.amin(self.buffer)

        mean = np.mean(self.buffer)
        std = np.std(self.buffer)

        buffer_no_outliers = self.buffer[((mean - 1 * std) <= self.buffer) & (self.buffer <= (mean + 1 * std))]
        self.buffer_max_no_outliers = np.amax(buffer_no_outliers)
        self.buffer_min_no_outliers = np.amin(buffer_no_outliers)

        self.buffer_max_log = np.log10(self.buffer_max)
        self.buffer_min_log = np.log10(self.buffer_min)
        self.buffer_max_no_outliers_log = np.log10(self.buffer_max_no_outliers)
        self.buffer_min_no_outliers_log = np.log10(self.buffer_min_no_outliers)


class _Fixed:
    def __init__(self):
        self.max: float
        self.min: float

    def set_statistics(self, min, max):
        self.min = min
        self.max = max


class OnlineNormalizationBuffer:
    def __init__(self, max_sizes):
        self.channel_state_information = _Buffer(max_size=max_sizes[0])
        self.power = _Fixed()
