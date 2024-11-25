from models import MSTpp
from tests import mrae
from datasets import NITRE_2022
import numpy as np

# TODO: make it a cli tool and take in arguments
all_datasets = [NITRE_2022.NITRE_2022(dataset_path="datasets/NITRE_2022")]

# for dataset in all_datasets:
#    dataset.download_dataset()


all_models = [
    MSTpp.MST_Plus_Plus(model_path="models/MSTpp/mst_plus_plus.pth"),
]
all_tests = [
    mrae.MRAE(),
]
for dataset in all_datasets:
    for model in all_models:
        model_predictions = []
        model.get_model()
        for test in all_tests:
            result = test.run_test(model, dataset)
            print(f"Test: {test.get_name()} Result: {result}")
            exit()
            # test.run_test(model, model_predictions)
        model.unload_model()
