from base.base_test import ScoreTest
from base.base_model import BaseModel
from base.base_dataset import BaseDataset
from tests.mse import MSE

import numpy as np
import time


class PSNR(ScoreTest):
    def __init__(self):
        super().__init__("PSNR")
        self.time = None

    def run_test(self, model: BaseModel, dataset: BaseDataset):
        psnr_values = []
        start_time = time.time()
        PEAK_NOISE = 255
        for img, validation_img in dataset.get_next_img():
            model_prediction = model.predict(img)
            assert model_prediction.shape == validation_img.shape
            mse = MSE.COMPUTE_MSE(validation_img, model_prediction)
            psnr = 10 * np.log10((PEAK_NOISE**2) / mse)
            psnr_values.append(psnr)
        self.time = time.time() - start_time
        self.results_per_image = psnr_values
