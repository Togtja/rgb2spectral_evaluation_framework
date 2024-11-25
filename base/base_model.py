from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path

    def get_name(self):
        return self.name

    @abstractmethod
    def get_model(self):
        # Load the model from the model_path
        # use this method to load the model, instead of loading it in the constructor
        pass

    @abstractmethod
    def unload_model(self):
        # Unload the model from memory
        pass

    @abstractmethod
    def predict(self, data):
        # Predict the output of the model
        pass
