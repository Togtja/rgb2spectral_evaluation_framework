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

    # Optional preprocess method (some datasets might need to preprocess the data to make it fit the 31 bands)
    def preprocess(self):
        print(f"No preprocess method implemented for {self.name}")
        return 1

    @abstractmethod
    # This method should return the next image in the dataset and the corresponding ground truth
    # The shape of the image should be (3, H, W) and the shape of the ground truth should be (31, H, W)
    def get_next_img(self):
        pass
