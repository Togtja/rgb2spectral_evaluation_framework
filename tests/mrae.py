from base.base_test import BaseTest
from base.base_model import BaseModel
from base.base_dataset import BaseDataset

import numpy as np


class MRAE(BaseTest):
    def __init__(self):
        super().__init__("MRAE")

    def run_test(self, model: BaseModel, dataset: BaseDataset):
        average_error = 0
        all_mrae = []
        model.get_model()
        for img, validation_img in dataset.get_next_img():
            model_prediction = model.predict(img)
            error = np.abs((model_prediction - validation_img) / validation_img)
            mare = np.mean(error)
            all_mrae.append(mare)
            average_error += mare
        average_error /= len(all_mrae)
        print(
            f"For model {model.get_name()} the average error is {average_error} in the {self.get_name()} test"
        )
        return average_error, all_mrae
