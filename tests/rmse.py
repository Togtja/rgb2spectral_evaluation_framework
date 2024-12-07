from base.base_test import ScoreTest
from base.base_model import BaseModel
from base.base_dataset import BaseDataset
from tests.mse import MSE

import numpy as np
import time


class RMSE(ScoreTest):
    def __init__(self):
        super().__init__("RMSE")

    def run_test(self, model: BaseModel, dataset: BaseDataset, result_type):
        all_rmse = []
        start_time = time.time()
        for img, validation_img in dataset.get_next_img():
            model_prediction = model.predict(img)
            assert model_prediction.shape == validation_img.shape
            mse = MSE.COMPUTE_MSE(validation_img, model_prediction)
            rmse = np.sqrt(mse)
            all_rmse.append(rmse)
        self.time = time.time() - start_time
        self.results_per_image = all_rmse
