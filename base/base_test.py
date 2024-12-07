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

    def correct_result_type(self, result_type):
        if result_type == "mean":
            return np.mean(self.results_per_image)
        elif result_type == "median":
            return np.median(self.results_per_image)
        elif result_type == "average":
            return np.average(self.results_per_image)
        else:
            print("Invalid result type, returning full results")
            return self.results_per_image

    @abstractmethod
    def run_test(self, model: BaseModel, dataset: BaseDataset, result_type="mean"):
        pass

    def get_results(self):
        results = self.correct_result_type("average")
        return results, self.results_per_image

    def get_time(self):
        return self.time


class VisualTest(BaseTest, ABC):
    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def plot_visuals(self):
        pass
