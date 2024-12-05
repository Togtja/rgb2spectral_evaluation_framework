from abc import ABC, abstractmethod
from .base_model import BaseModel
from .base_dataset import BaseDataset


class BaseTest(ABC):
    def __init__(self, name):
        self.name = name
        self.time = None

    def get_name(self):
        return self.name

    @abstractmethod
    def run_test(self, model: BaseModel, dataset: BaseDataset):
        pass

    @abstractmethod
    def get_time(self):
        return self.time
