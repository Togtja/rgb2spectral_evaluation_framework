from abc import ABC, abstractmethod
from .base_model import BaseModel


class BaseTest(ABC):
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    @abstractmethod
    def run_test(self, model: BaseModel, dataset):
        pass
