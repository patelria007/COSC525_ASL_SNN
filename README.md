# CSNN4ASLDVS

This is a UTK COSC525 Deep Learning final project. We created a Convolutional Spiking Neural Network (CSNN) for the neuromorphic [ASL-DVS dataset](https://github.com/PIX2NVS/NVS2Graph/tree/master) devleloped by Yin Bi et al. This repository accompanies our final report 
_Using CSNNs to Perform Event-based Data Processing & Classification on ASL-DVS_.


## Authors
- [Ria Patel](https://github.com/patelria007)
- [Sujit Tripathy](https://github.com/Suj8trip8)
- [Zachary Sublett](https://github.com/zachUTK)
- [Seoyoung An](https://github.com/seoyoung16)
- [Riya Patel](https://github.com/rpatel90)

## Requirements
**Anaconda (Conda 22.11.1)**:
- Use `conda` environments 
- `environment.yml`: creates environment, run `conda env create -f environment.yml`. 

The following frameworks are required to run the CSNN. These are included in the `environment.yml` file.
- PyTorch
- Sinabs
- Tonic


## Dataset

Access to the [converted ASL dataset](https://drive.google.com/drive/folders/1eVItyrWI0HDg7sj4hEpfh3w0yUgdz70j?usp=drive_link) we generated from `.aedat` (AEDAT 2.0) to `.aedat4` (AEDAT 4.0) using the DV-Processing software. 

Access to the [preprocessed ASL-DVS subset](https://drive.google.com/file/d/1Xd7xBqTR4KRLAyJYcWkxIJ3DVVw3kl56/view?usp=sharing) of the ASL-DVS dataset.

The data is in the form of 5 `.zip` files, one for each of the subjects used in data collection. The authors of the original code repository, [NVS2Graph](https://github.com/PIX2NVS/NVS2Graph/tree/master), developed this dataset to accompany their code.

### Dataset Characteristics:
* 24 classes (`A-Y`, excludes `J`)
 ![f](https://github.com/patelria007/COSC525_ASL_SNN/assets/91634833/3ce1998e-37fc-417a-a190-4935d9da2491)
* 5 subjects used to pose handshapes
* 4,200 samples **_per letter_** (100,800 total samples)
    * Each sample lasts 100 milliseconds 
* Recorded with iniLabs DAVIS240c NVS camera
    * Office with constant illumination, low environmental noise

### Dataset Exploration
`utils/data_explore.ipynb`: Jupyter Notebook that shows dataset characteristics such as sampling frequencies amongst all subjects, events per batch sample, etc. 

### Dataset Preprocessing
`utils/data_preprocess.ipynb`: Jupyter Notebook that generates the preprocessed training and validation datasets in the form of `.bin` Numpy files. These will easily load into the PyTorch DataLoader that will then input into our CSNN. 

## CSNN Model Training
