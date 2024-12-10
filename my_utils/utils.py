# Download files from a google drive

import gdown
import zipfile
import requests
import cv2
import numpy as np


def download_file_from_google_drive(url, output):
    gdown.download(url, output, quiet=False, fuzzy=True)


def unzip_file(file, output_folder):
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(output_folder)


def download_file(url, output):
    if not url.startswith(("http://", "https://")):
        raise ValueError(
            f"Invalid URL '{url}': No scheme supplied. Perhaps you meant 'https://{url}'?"
        )
    print(f"Downloading {url} to {output}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


def read_img(img_path):
    # Read an image from a file
    img = cv2.imread(img_path)
    # Ensure the image is in RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img)
    # Normalize the image
    img = (img - img.min()) / (img.max() - img.min())
    return img
