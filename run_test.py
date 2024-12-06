from base.base_test import VisualTest, ScoreTest
from models import App, MSTpp
from tests import mrae, psnr, rmse
from datasets import NTIRE_2022, Generic_Matlab, SIDQ

# TODO: make it a cli tool and take in arguments
all_datasets = [
    SIDQ.SIDQ(dataset_path="datasets/SIDQ"),
    NTIRE_2022.NTIRE_2022(dataset_path="datasets/NTIRE_2022"),
    #    Generic_Matlab.Generic_Matlab(dataset_path="datasets/Generic_Matlab"),
]
all_models = [
    App.App(model_path="models/A++/model_a_plus_plus_retrain.pkl"),
    MSTpp.MST_Plus_Plus(model_path="models/MSTpp/mst_plus_plus.pth"),
]
all_tests = [
    mrae.MRAE(),
    psnr.PSNR(),
    rmse.RMSE(),
]
for dataset in all_datasets:
    for model in all_models:
        model_predictions = []
        model.get_model()
        for test in all_tests:
            print(f"Running test: {test.get_name()} on model: {model.get_name()}")
            result = test.run_test(model, dataset)
            print(f"Test: {test.get_name()} Result: {result}")
            print(f"Test: {test.get_name()} Time: {test.get_time()}")
        model.unload_model()

# TODO: Write the results to a file
