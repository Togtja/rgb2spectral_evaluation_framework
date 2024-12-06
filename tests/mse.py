from base.base_test import ScoreTest
from base.base_model import BaseModel
from base.base_dataset import BaseDataset

import numpy as np
import time


class MSE(ScoreTest):
    def __init__(self):
        super().__init__("MSE")

    def COMPUTE_MSE(a, b):
        return np.power((a - b) ** 2).mean()

    def run_test(self, model: BaseModel, dataset: BaseDataset, result_type):
        all_mse = []
        start_time = time.time()
        for img, validation_img in dataset.get_next_img():
            model_prediction = model.predict(img)
            assert model_prediction.shape == validation_img.shape
            mse = np.mean((validation_img - model_prediction) ** 2)
            all_mse.append(mse)
        self.time = time.time() - start_time
        self.results = self.correct_result_type(all_mse, result_type)

    def get_time(self):
        return self.time
