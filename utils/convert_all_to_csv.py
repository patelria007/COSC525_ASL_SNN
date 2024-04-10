import os
from tqdm import tqdm
import dv_processing as dv

DATA = "../data"
CSV = f"{DATA}/csv"
AEDAT = f"{DATA}/aedat"
TARBALLS = f"{DATA}/tarballs"

# Create necessary directories
if not os.path.isdir(DATA):
    os.mkdir(DATA)
if not os.path.isdir(TARBALLS):
    os.mkdir(TARBALLS)
if not os.path.isdir(CSV):
    os.mkdir(CSV)
if not os.path.isdir(AEDAT):
    os.mkdir(AEDAT)

# Move all zipped files to data
#os.system(F"move {DATA}/*.zip {TARBALLS}")

# Unzip all tarballss
for i in tqdm(os.listdir(TARBALLS)):
    if i.endswith(".zip"):
        FILE = f"{TARBALLS}/{i}"
        os.system(f"tar -xf {FILE} -C {AEDAT}")
        if "__MACOSX" in os.listdir(AEDAT):
            os.system(f"rmdir {AEDAT}/__MACOSX")


for i in tqdm(sorted(os.listdir(AEDAT))):
    SUBJECT = f"{AEDAT}/{i}"
    if not os.path.isdir(SUBJECT):
        os.mkdir(SUBJECT)
    
    EXP = f"{CSV}/{i}"
    if not os.path.isdir(EXP):
        os.mkdir(EXP)

    for j in os.listdir(SUBJECT):
        FILE = f"{SUBJECT}/{j}"
        os.system(f"python export_aedat4_to_csv.py --file {FILE} --dir {EXP}")