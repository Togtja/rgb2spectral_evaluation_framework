import base.base_dataset as base_dataset
import os
import my_utils.utils as utils
import h5py
import scipy.io
import numpy as np


class Generic_Matlab(base_dataset.BaseDataset):
    def __init__(self, dataset_path):
        super().__init__("Generic_Matlab", dataset_path)
        self.namefiles = [
            "Indian_pines",
            "Salinas",
            "SalinasA",
            "Pavia",
            "PaviaU",
            "KSC",
            "Botswana",
        ]

    def download_dataset(self):
        DATASET_PATH = f"./datasets/{self.name}"
        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)
        else:
            print(f"Dataset {self.name} already exists. Skipping download.")
            return 0
        # Download the dataset from the internet
        simple_urls_image = [
            "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
            "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
            "https://www.ehu.eus/ccwintco/uploads/1/1a/SalinasA_corrected.mat",
            "https://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat",
            "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
            "http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat",
            "http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat",
        ]
        simple_url_gt = [
            "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
            "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
            "https://www.ehu.eus/ccwintco/uploads/a/aa/SalinasA_gt.mat",
            "https://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat",
            "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
            "http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat",
            "http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat",
        ]

        for img_url, gt_url, name in zip(
            simple_urls_image, simple_url_gt, self.namefiles
        ):
            gt = f"{name}_gt.mat"
            rgb = f"{name}_rgb.mat"
            print(f"Downloading {name}..., {img_url}, {gt_url}")

            utils.download_file(img_url, f"{DATASET_PATH}/{rgb}")
            utils.download_file(gt_url, f"{DATASET_PATH}/{gt}")

    def get_next_img(self):
        for name in self.namefiles:
            img = load_matlab_file(f"{self.dataset_path}/{name}_rgb.mat")
            gt = load_matlab_file(f"{self.dataset_path}/{name}_gt.mat")
            print(f"Loaded {name} with shape: {img.shape}, {gt.shape}")

            yield img, gt


def load_matlab_h5f5_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit()
        return None
    try:
        with h5py.File(file_path, "r") as f:
            # Print the items in the file
            print("Items in the file: ", list(f.keys()))
            # Assuming 'cube' is the dataset you want to read
            data = f["cube"][:]
            return np.array(data)
    except OSError as e:
        print(f"Error opening file {file_path}: {e}")
        return None
    exit()


def load_matlab_file(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    try:
        mat = scipy.io.loadmat(file_path)
        # Print the items in the file
        print("Items in the file: ", list(mat.keys()))
        # Assuming 'cube' is the dataset you want to read
        data = mat.get(list(mat.keys())[-1])
        if data is None:
            print(f"'cube' not found in {file_path}")
            return None
        return np.array(data)
    except Exception as e:
        print(f"Error opening file {file_path}: {e}")
        return None
