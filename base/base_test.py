from abc import ABC, abstractmethod
from .base_model import BaseModel
from .base_dataset import BaseDataset
import numpy as np


class BaseTest(ABC):
    def __init__(self, name):
        self.name = name
        self.time = None

    def get_name(self):
        return self.name

    @abstractmethod
    def get_time(self):
        return self.time


class ScoreTest(BaseTest, ABC):
    def __init__(self, name):
        super().__init__(name)
        # The array of results for each image
        self.results_per_image = None
        # Time in seconds to run the test
        self.time = None

    def result_pooling(self, pooling):
        if pooling == "mean":
            return float(np.mean(self.results_per_image))
        elif pooling == "median":
            return float(np.median(self.results_per_image))
        elif pooling == "average":
            return float(np.average(self.results_per_image))
        else:
            print("Invalid result type, returning full results")
            return self.results_per_image

    @abstractmethod
    def run_test(self, model: BaseModel, dataset: BaseDataset):
        pass

    def get_results(self, pooling_method="average"):
        results = self.result_pooling(pooling_method)
        return results, self.results_per_image

    def get_time(self):
        return self.time


class VisualTest(BaseTest, ABC):
    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def plot_visuals(self):
        pass
