from base.base_dataset import BaseDataset
import my_utils.utils as utils
import os
import h5py
import numpy as np
import cv2
import shutil


class CAVE(BaseDataset):
    def __init__(self, dataset_path):
        super().__init__("CAVE", dataset_path)

    def download_dataset(self):
        DATASET_PATH = f"./datasets/{self.name}"
        # if not os.path.exists(DATASET_PATH):
        #    os.makedirs(DATASET_PATH)
        # else:
        #    print(f"Dataset {self.name} already exists. Skipping download.")
        #    return 0
        ## Overwrite the dataset path as we are downloading the dataset
        # self.dataset_path = DATASET_PATH
        # URL = "https://cave.cs.columbia.edu/old/databases/multispectral/zip/complete_ms_data.zip"
        # utils.download_file(URL, f"{DATASET_PATH}/{self.name}.zip")
        # utils.unzip_file(f"{DATASET_PATH}/{self.name}.zip", f"{DATASET_PATH}/")
        # os.remove(f"{DATASET_PATH}/{self.name}.zip")
        #
        # Organize the dataset
        # Each folder is doubled up, remove the duplicates
        for folder in os.listdir(f"{DATASET_PATH}"):
            if os.path.isdir(f"{DATASET_PATH}/{folder}/{folder}"):

                os.rename(
                    f"{DATASET_PATH}/{folder}/{folder}",
                    f"{DATASET_PATH}/{folder}",
                )

    def preprocess(self):
        # Preprocess the dataset
        os.makedirs(f"{self.dataset_path}/gt", exist_ok=True)
        os.makedirs(f"{self.dataset_path}/rgb", exist_ok=True)
        for folder in os.listdir(f"{self.dataset_path}"):
            if (
                not os.path.isdir(f"{self.dataset_path}/{folder}")
                or folder == "gt"
                or folder == "rgb"
            ):
                continue
            # open each folder_01.png to folder_31.png and make into a single array and save as .mat
            img_31 = np.zeros((31, 512, 512))
            for i in range(1, 32):
                img_path = f"{self.dataset_path}/{folder}/{folder}_{i:02d}.png"
                if not os.path.exists(img_path):
                    continue
                img = utils.read_img(img_path)
                # Ensure the image is grayscale by checking if all channels are equal
                assert np.all(img[:, :, 0] == img[:, :, 1]) and np.all(
                    img[:, :, 1] == img[:, :, 2]
                )
                img = img[:, :, 0]
                img_31[i - 1] = img
                os.remove(img_path)
            utils.save_mat(f"{self.dataset_path}/gt/{folder}.mat", img_31)
            # Load the BMP file and save as .png
            base_name = folder.replace("_ms", "")
            img = cv2.imread(f"{self.dataset_path}/{folder}/{base_name}_RGB.bmp")
            cv2.imwrite(f"{self.dataset_path}/rgb/{base_name}.png", img)
            os.remove(f"{self.dataset_path}/{folder}/{base_name}_RGB.bmp")
            # Remove the folder
            shutil.rmtree(f"{self.dataset_path}/{folder}")

    def get_next_img(self):
        for folder in os.listdir(f"{self.dataset_path}/gt"):
            gt = h5py.File(f"{self.dataset_path}/gt/{folder}", "r")["data"][:]
            rgb = utils.read_img(
                f"{self.dataset_path}/rgb/{folder.replace('_ms.mat', '.png')}"
            )
            rgb = rgb.transpose(2, 0, 1)
            yield rgb, gt
