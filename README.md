# COSC525_ASL_SNN
UTK COSC525 - Deep Learning class project - Spiking Neural Network for neuromorphic ASL dataset

## Dataset
Access the converted ASL dataset from `.aedat` (AEDAT 2.0) to `.aedat4` (AEDAT 4.0) here: https://drive.google.com/drive/folders/1eVItyrWI0HDg7sj4hEpfh3w0yUgdz70j?usp=drive_link 

The data is in the form of 5 `.zip` files, one for each of the subjects used in data collection. The original dataset is found here: https://github.com/PIX2NVS/NVS2Graph/tree/master.

### Dataset Characteristics:
* 24 classes (`A-Y`, excludes `J`)
* 5 subjects used to pose handshapes
* 4,200 samples **_per letter_** (100,800 total samples)
    * Each sample lasts 100 milliseconds 
* Recorded with iniLabs DAVIS240c NVS camera
    * Office with constant illumination, low envioronmental noise

