from base.base_test import ScoreTest
from base.base_model import BaseModel
from base.base_dataset import BaseDataset

import numpy as np
import time


class MRAE(ScoreTest):
    def __init__(self):
        super().__init__("MRAE")

    def run_test(self, model: BaseModel, dataset: BaseDataset):
        all_mrae = []
        start_time = time.time()
        for img, validation_img in dataset.get_next_img():
            mask = validation_img == 0
            if mask.any():
                validation_wo_zeros = validation_img.copy()
                validation_wo_zeros[mask] = 1e-8  # Avoid division by zero
            else:
                validation_wo_zeros = validation_img
            model_prediction = model.predict(img)
            assert model_prediction.shape == validation_img.shape
            error = np.abs((validation_img - model_prediction) / validation_wo_zeros)
            mare = np.mean(error)
            all_mrae.append(mare)
        self.time = time.time() - start_time
        self.results_per_image = all_mrae
