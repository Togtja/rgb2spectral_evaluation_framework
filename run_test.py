import os
import json

# Set the environment variable to allow multiple OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from base.base_test import VisualTest, ScoreTest
from models import App, MSTpp
from tests import mrae, psnr, rmse
from datasets import NTIRE_2022, SIDQ

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

# Remove tests from the results if you want to rerun them but not the others
remove_tests = []
USE_OLD_RESULTS = True
if USE_OLD_RESULTS and os.path.exists("results.json"):
    with open("results.json", "r") as f:
        results = json.load(f)
    for remove_test in remove_tests:
        for dataset in results:
            for model in results[dataset]:
                if remove_test in results[dataset][model]:
                    del results[dataset][model][remove_test]
                else:
                    print(
                        f"Test {remove_test} not found in results for {dataset} {model}"
                    )
else:
    results = {}


DOWNLOAD_DATASETS = True
if DOWNLOAD_DATASETS:
    for dataset in all_datasets:
        dataset.download_dataset()
    print("Downloaded all datasets")
    exit(0)

PREPROCESS_DATASETS = True
if PREPROCESS_DATASETS:
    for dataset in all_datasets:
        dataset.preprocess()
    exit(0)


def save_results():
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)


for dataset in all_datasets:
    if dataset.get_name() not in results:
        results[dataset.get_name()] = {}
    for model in all_models:
        if model.get_name() not in results[dataset.get_name()]:
            results[dataset.get_name()][model.get_name()] = {}
        model.get_model()
        for test in all_tests:
            # Skip if the test has already been run
            if test.get_name() in results[dataset.get_name()][model.get_name()]:
                continue
            print(
                f"Running test: {test.get_name()} on model: {model.get_name()} with dataset: {dataset.get_name()}"
            )
            test.run_test(model, dataset, "average")
            # if test.type() == TestType.SCORE:
            if isinstance(test, ScoreTest):
                result, result_per_iamge = test.get_results()
                print(f"Test: {test.get_name()} Result: {result}")
                print(f"Test: {test.get_name()} Time: {test.get_time()}")
                results[dataset.get_name()][model.get_name()][test.get_name()] = {
                    "result": result,
                    "result_type": "average",
                    "time": test.get_time(),
                }
                save_results()
            elif isinstance(test, VisualTest):
                test.plot_visuals()
            else:
                print(f"Test type not recognized: {type(test).__name__}")
            # TODO Add the results per image to the results dictionary
        model.unload_model()
