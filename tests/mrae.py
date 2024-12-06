from base.base_test import BaseTest
from base.base_model import BaseModel
from base.base_dataset import BaseDataset

import numpy as np
import time


class MRAE(ScoreTest):
    def __init__(self):
        super().__init__("MRAE")

    def run_test(self, model: BaseModel, dataset: BaseDataset, result_type):
        all_mrae = []
        start_time = time.time()
        for img, validation_img in dataset.get_next_img():
            model_prediction = model.predict(img)
            error = np.abs((model_prediction - validation_img) / validation_img)
            if np.isnan(error).any() or np.isinf(error).any():
                # print(f"Error: {error}")
                continue

            mare = np.mean(error)
            all_mrae.append(mare)
        self.time = time.time() - start_time
        self.results = self.correct_result_type(all_mrae, result_type)

    def get_time(self):
        return self.time
