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

    def correct_result_type(self, full_results, result_type):
        if result_type == "mean":
            return np.mean(full_results)
        elif result_type == "median":
            return np.median(full_results)
        elif result_type == "average":
            return np.average(full_results)
        else:
            print("Invalid result type, returning full results")
            return full_results

    @abstractmethod
    def run_test(self, model: BaseModel, dataset: BaseDataset, result_type="mean"):
        pass

    @abstractmethod
    def get_results(self):
        pass


class VisualTest(BaseTest, ABC):
    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def plot_visuals(self):
        pass
