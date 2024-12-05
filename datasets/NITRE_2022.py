from base.base_dataset import BaseDataset
import os

import my_utils.utils as utils

import numpy as np
import h5py
import cv2


class NITRE_2022(BaseDataset):
    def __init__(self, dataset_path):
        super().__init__("NITRE_2022", dataset_path)
        self.folders = [
            "Train_Spec",
            "Train_RGB",
            "validation_spectral",
            "validation_rgb",
            "test_rgb",
        ]

    def download_dataset(self):
        DATASET_PATH = f"./datasets/{self.name}"
        # Download the dataset from the internet
        # if not os.path.exists(DATASET_PATH):
        #    os.makedirs(DATASET_PATH)
        # else:
        #    print(f"Dataset {self.name} already exists. Skipping download.")
        #    return 0
        TRAIN_SPECTRAL_URL = (
            "https://drive.google.com/file/d/1FQBfDd248dCKClR-BpX5V2drSbeyhKcq/view"
        )
        TRAIN_RGB_URL = (
            "https://drive.google.com/file/d/1A4GUXhVc5k5d_79gNvokEtVPG290qVkd/view"
        )
        VALIDATION_SPECTRAL_URL = (
            "https://drive.google.com/file/d/12QY8LHab3gzljZc3V6UyHgBee48wh9un/view"
        )
        VALIDATION_RGB_URL = (
            "https://drive.google.com/file/d/19vBR_8Il1qcaEZsK42aGfvg5lCuvLh1A/view"
        )
        # TEST_RGB_URL = (
        #    "https://drive.google.com/file/d/1A5309Gk7kNFI-ORyADueiPOCMQNTA7r5/view"
        # )

        # utils.download_file_from_google_drive(
        #    TRAIN_SPECTRAL_URL, f"{DATASET_PATH}/train_spectral.zip"
        # )
        # utils.download_file_from_google_drive(
        #    TRAIN_RGB_URL, f"{DATASET_PATH}/train_rgb.zip"
        # )
        # utils.download_file_from_google_drive(
        #    VALIDATION_SPECTRAL_URL, f"{DATASET_PATH}/validation_spectral.zip"
        # )
        utils.download_file_from_google_drive(
            VALIDATION_RGB_URL, f"{DATASET_PATH}/validation_rgb.zip"
        )
        # utils.download_file_from_google_drive(
        #    TEST_RGB_URL, f"{DATASET_PATH}/test_rgb.zip"
        # )

        # Unzip the files
        # For each file in the folder, unzip it
        for file in os.listdir(DATASET_PATH):
            if file.endswith(".zip"):
                zip_file = DATASET_PATH + "/" + file
                utils.unzip_file(zip_file, DATASET_PATH)
                os.remove(zip_file)
        VALID_RGB = "Valid_RGB"
        VALID_SPEC = "Valid_Spec"

        self.dataset_path = DATASET_PATH

        return 0

    def get_next_img(self):
        # Get files in folder
        train_rgb_directory = self.dataset_path + "/Train_RGB"
        train_spec_directory = self.dataset_path + "/Train_Spec"
        for file in os.listdir(train_rgb_directory):
            matlab_file = file.replace(".jpg", ".mat")
            matlab_file = train_spec_directory + "/" + matlab_file
            print(f"Processing {matlab_file}")
            try:
                valid_img = load_matlab_file(matlab_file)
                # Transpose valid_img to match model_prediction shape
                valid_img = np.transpose(valid_img, (0, 2, 1))
                img = cv2.imread(train_rgb_directory + "/" + file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                yield img, valid_img
            except OSError as e:
                print(f"Error processing {matlab_file}: {e}")
                continue


def load_matlab_file(file_path):
    with h5py.File(file_path, "r") as f:
        # Assuming 'cube' is the dataset you want to read
        data = f["cube"][:]
        return np.array(data)
