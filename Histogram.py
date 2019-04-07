import numpy as np
import matplotlib.pyplot as plt

NUM_BUCKETS = 10


class Histogram:

    def __init__(self, dimension_data, num_buckets=NUM_BUCKETS):
        data_len = len(dimension_data)
        self.min = np.min(dimension_data)
        max = np.max(dimension_data)
        self.bucket_size = (max - self.min + 0.001) / num_buckets
        self.hist = np.zeros(num_buckets)
        for d in dimension_data:
            index = int((d - self.min) // self.bucket_size)
            self.hist[index] = self.hist[index] + 1
        for i in range(num_buckets):
            self.hist[i] = self.hist[i] / data_len

    def get_min(self):
        return self.min

    def get_histogram(self):
        return self.hist

    def get_bucket_size(self):
        return self.bucket_size

    def get_bucket(self, value):
        index = int((value - self.min) // self.bucket_size)
        return max(0, min(index, len(self.hist) - 1))
