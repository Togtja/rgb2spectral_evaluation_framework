from base.base_dataset import BaseDataset
import os

import my_utils.utils as utils

import numpy as np
import h5py


class NTIRE_2022(BaseDataset):
    def __init__(self, dataset_path):
        super().__init__("NTIRE_2022", dataset_path)

    def download_dataset(self):
        DATASET_PATH = self.dataset_path

        # Download the dataset from the internet
        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)
        else:
            print(f"Dataset {self.name} already exists. Skipping download.")
            return 0
        google_downloads = {
            "TRAIN_SPECTRAL": "https://drive.google.com/file/d/1FQBfDd248dCKClR-BpX5V2drSbeyhKcq/view",
            "TRAIN_RGB": "https://drive.google.com/file/d/1A4GUXhVc5k5d_79gNvokEtVPG290qVkd/view",
            "VALIDATION_SPECTRAL": "https://drive.google.com/file/d/12QY8LHab3gzljZc3V6UyHgBee48wh9un/view",
            "VALIDATION_RGB": "https://drive.google.com/file/d/19vBR_8Il1qcaEZsK42aGfvg5lCuvLh1A/view",
            "314_PATCH": "https://drive.google.com/file/d/1DOsjSMC9jHMBF0KzHTipbjwZbqyVF20K/view",
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
            elif "314_PATCH" in file_name:
                continue
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
            # Apply 314_PATCH
            os.remove(f"{train_spec_directory}/ARAD_1K_0314.mat")
            os.rename(
                f"{DATASET_PATH}/314_PATCH.zip",  # Not actually a zip file
                f"{train_spec_directory}/ARAD_1K_0314.mat",
            )
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
                img = utils.read_img(train_rgb_directory + "/" + file)
                img = np.transpose(img, [2, 0, 1])
                yield np.ascontiguousarray(img), np.ascontiguousarray(valid_img)
            except OSError as e:
                print(f"Error processing {matlab_file}: {e}")
                continue


def load_matlab_file(file_path):
    with h5py.File(file_path, "r") as f:
        # Assuming 'cube' is the dataset you want to read
        # data = f["cube"][:]
        hyper = np.float32(np.array(f["cube"]))
    # data = np.float32(np.array(data))
    return np.transpose(hyper, [0, 2, 1])
