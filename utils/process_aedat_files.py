"""
Process AEDAT files.

This script processes AEDAT files by moving zipped files, 
unzipping tarballs, and converting AEDAT4.0 files to CSV format.

Args:
    --data (str): Path to the data directory containing AEDAT files.

Usage:
    python process_aedat_files.py --data /path/to/data

The script expects the following directory structure:
- data/
  - tarballs/ (to store zipped files)
  - aedat/ (to store unzipped AEDAT files)
  - csv/ (to store CSV files)

It performs the following steps:
1. Moves zipped files from the data directory to the tarballs directory.
2. Unzips tarball files in the tarballs directory to the aedat directory.
3. Converts AEDAT files in the aedat directory to CSV format and saves them in the csv directory.

"""

import os
from tqdm import tqdm
import shutil # to remove folders that are not empty
import argparse

def create_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

def move_zipped_files(source, destination):
    zip_files = [f for f in os.listdir(source) if f.endswith(".zip")]
    for file in tqdm(zip_files, desc="Moving zipped files"):
        shutil.move(os.path.join(source, file), destination)

def unzip_tarballs(source, destination):
    zip_files = [f for f in os.listdir(source) if f.endswith(".zip")]
    for file in tqdm(zip_files, desc="Unzipping tarballs"):
        file_path = os.path.join(source, file)
        os.system(f"tar -xf {file_path} -C {destination}")
        if "__MACOSX" in os.listdir(destination):
            shutil.rmtree(os.path.join(destination, "__MACOSX"))

def process_aedat_files(aedat_directory, csv_directory):
    for subject in tqdm(sorted(os.listdir(aedat_directory)), desc="Converting AEDAT to CSV"):
        subject_dir = os.path.join(aedat_directory, subject)
        if not os.path.isdir(subject_dir):
            os.mkdir(subject_dir)

        exp_dir = os.path.join(csv_directory, subject)
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)

        for file in os.listdir(subject_dir):
            if file.endswith(".aedat4"):
                file_path = os.path.join(subject_dir, file)
                os.system(f"python export_aedat4_to_csv.py --file {file_path} --dir {exp_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process AEDAT files.')
    parser.add_argument('--data', type=str, required=True, help='Path to the data directory')
    args = parser.parse_args()

    DATA = args.data
    CSV = os.path.join(DATA, "csv")
    AEDAT = os.path.join(DATA, "aedat")
    TARBALLS = os.path.join(DATA, "tarballs")

    create_directory(DATA)
    create_directory(TARBALLS)
    create_directory(CSV)
    create_directory(AEDAT)

    move_zipped_files(DATA, TARBALLS)
    unzip_tarballs(TARBALLS, AEDAT)
    process_aedat_files(AEDAT, CSV)
