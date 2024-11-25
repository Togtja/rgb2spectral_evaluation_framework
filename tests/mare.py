from ..base.base_test import BaseTest
from ..base.base_model import BaseModel

import numpy as np


class MARE(BaseTest):
    def __init__(self):
        super().__init__("MARE")

    def run_test(self, model: BaseModel, dataset):
        print("Running test: ", self.get_name() + " on model: ", model.get_name())
        # Add test code here

        for img in dataset:
            rgb = np.load(img)

        print("Test completed")
