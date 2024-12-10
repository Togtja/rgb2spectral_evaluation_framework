from base.base_dataset import BaseDataset
import my_utils.utils as utils
import os
import h5py
import numpy as np
from sklearn.decomposition import PCA


class SIDQ(BaseDataset):
    def __init__(self, dataset_path):
        super().__init__("SIDQ", dataset_path)

    def download_dataset(self):
        DATASET_PATH = f"./datasets/{self.name}"
        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)
        else:
            print(f"Dataset {self.name} already exists. Skipping download.")
            return 0
        # Overwrite the dataset path as we are downloading the dataset
        self.dataset_path = DATASET_PATH
        URL = "http://folk.ntnu.no/mariupe/sidq.zip"
        utils.download_file(URL, f"{DATASET_PATH}/{self.name}.zip")
        utils.unzip_file(f"{DATASET_PATH}/{self.name}.zip", f"{DATASET_PATH}/")
        os.remove(f"{DATASET_PATH}/{self.name}.zip")

    def preprocess(self):
        if os.path.exists(f"{self.dataset_path}/preprocessed_31_bands"):
            print("Preprocessed folder already exists. Skipping preprocessing.")
            return 0
        os.makedirs(f"{self.dataset_path}/preprocessed_31_bands", exist_ok=True)
        gt_images_path = self.dataset_path + "/Original Images"
        for img_name in os.listdir(gt_images_path):
            output_name = img_name.split(".")[0] + ".npy"
            if os.path.exists(
                f"{self.dataset_path}/preprocessed_31_bands/{output_name}"
            ):
                continue
            gt = pca_160_to_31(f"{gt_images_path}/{img_name}")
            np.save(
                f"{self.dataset_path}/preprocessed_31_bands/{output_name}.npy",
                gt,
            )

    def get_next_img(self):
        rgb_images_path = self.dataset_path + "/RGB renderings (for CIE D65)"
        gt_images_path = self.dataset_path + "/Original Images"
        os.listdir(rgb_images_path)
        for img_name in os.listdir(rgb_images_path):
            if img_name[0] == ".":
                continue
            base_name = img_name.split(".")[0].split("_")[0]
            # Check if preprocessed file exists
            if os.path.exists(
                f"{self.dataset_path}/preprocessed_31_bands/{base_name}.npy"
            ):
                gt = np.load(
                    f"{self.dataset_path}/preprocessed_31_bands/{base_name}.npy"
                )
            else:
                gt = pca_160_to_31(f"{gt_images_path}/{base_name}.mat")
            # Read as RGB
            img = utils.read_img(f"{rgb_images_path}/{img_name}")
            img = np.transpose(img, (2, 0, 1))
            yield img, gt


def pca_160_to_31(img_path):
    # Original spectral image is 160 bands from 410 to 1000 nm
    # We will reduce it to 31 bands using PCA
    # Original image shape is (C, H, W) where C is 160
    with h5py.File(img_path, "r") as f:
        img = f["hsi"][:]

    # Transpose to (H, W, C)
    img = np.transpose(img, (1, 2, 0))

    # Flatten the hyperspectral data
    hsi_flattened = img.reshape(-1, img.shape[-1])

    # Apply PCA
    pca = PCA(n_components=31)
    hsi_pca = pca.fit_transform(hsi_flattened)

    # Reshape back to (H, W, 31) and then transpose to (31, H, W)
    hsi_pca_reshaped = hsi_pca.reshape(img.shape[0], img.shape[1], 31)
    hsi_pca_reshaped = np.transpose(hsi_pca_reshaped, (2, 0, 1))
    # Normalize the data
    hsi_pca_reshaped = (hsi_pca_reshaped - hsi_pca_reshaped.min()) / (
        hsi_pca_reshaped.max() - hsi_pca_reshaped.min()
    )
    return hsi_pca_reshaped
