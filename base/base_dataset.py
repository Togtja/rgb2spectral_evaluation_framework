from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(self, name, dataset_path):
        self.name = name
        self.dataset_path = dataset_path

    def get_name(self):
        return self.name

    # Optional download method
    def download_dataset(self):
        # Download the dataset from the internet
        print(f"No download method implemented for {self.name}")
        return 1

    @abstractmethod
    def get_next_img(self):
        pass
