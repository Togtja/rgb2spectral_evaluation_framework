from base.base_dataset import BaseDataset
import os

import my_utils.utils as utils

import numpy as np
import h5py
import cv2


class NTIRE_2022(BaseDataset):
    def __init__(self, dataset_path):
        super().__init__("NTIRE_2022", dataset_path)

    def download_dataset(self):
        DATASET_PATH = self.dataset_path

        # Download the dataset from the internet
        # if not os.path.exists(DATASET_PATH):
        #    os.makedirs(DATASET_PATH)
        # else:
        #    print(f"Dataset {self.name} already exists. Skipping download.")
        #    return 0
        google_downloads = {
            "TRAIN_SPECTRAL": "https://drive.google.com/file/d/1FQBfDd248dCKClR-BpX5V2drSbeyhKcq/view",
            "TRAIN_RGB": "https://drive.google.com/file/d/1A4GUXhVc5k5d_79gNvokEtVPG290qVkd/view",
            "VALIDATION_SPECTRAL": "https://drive.google.com/file/d/12QY8LHab3gzljZc3V6UyHgBee48wh9un/view",
            "VALIDATION_RGB": "https://drive.google.com/file/d/19vBR_8Il1qcaEZsK42aGfvg5lCuvLh1A/view",
        }

        train_rgb_directory = DATASET_PATH + "/Train_RGB"
        train_spec_directory = DATASET_PATH + "/Train_Spec"
        os.makedirs(train_spec_directory, exist_ok=True)
        for file_name, url in google_downloads.items():
            zip_file = f"{DATASET_PATH}/{file_name}.zip"
            if not os.path.exists(zip_file):
                utils.download_file_from_google_drive(url, zip_file)
            # Unzip the files
            if "SPECTRAL" in file_name:
                utils.unzip_file(zip_file, train_spec_directory)
            else:
                utils.unzip_file(zip_file, DATASET_PATH)

            # Move all .jpg files in the validation folder to the Train_RGB folder
            if "VALIDATION_RGB" in file_name:
                validation_rgb_directory = DATASET_PATH + "/Valid_RGB"
                for root, dirs, files in os.walk(validation_rgb_directory):
                    for file in files:
                        if file.endswith(".jpg"):
                            os.rename(f"{root}/{file}", f"{train_rgb_directory}/{file}")
                os.rmdir(validation_rgb_directory)
        return 0

    def get_next_img(self):
        # Get files in folder
        train_rgb_directory = self.dataset_path + "/Train_RGB"
        train_spec_directory = self.dataset_path + "/Train_Spec"
        for file in os.listdir(train_rgb_directory):
            matlab_file = file.replace(".jpg", ".mat")
            matlab_file = train_spec_directory + "/" + matlab_file
            try:
                valid_img = load_matlab_file(matlab_file)
                # Transpose valid_img to match model_prediction shape

                valid_img = np.transpose(valid_img, (2, 1, 0))
                img = cv2.imread(train_rgb_directory + "/" + file, cv2.IMREAD_COLOR)
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
