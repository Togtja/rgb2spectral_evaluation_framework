# Download files from a google drive

import gdown
import zipfile


def download_file_from_google_drive(url, output):
    gdown.download(url, output, quiet=False, fuzzy=True)


def unzip_file(file, output_folder):
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(output_folder)
