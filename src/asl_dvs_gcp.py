from tonic.dataset import Dataset
from typing import Any, Callable, Optional, Tuple
import pandas as pd
import numpy as np
from google.cloud import storage

class ASLDVS(Dataset):
    """`ASL-DVS <https://github.com/PIX2NVS/NVS2Graph>`
    
    ASL-DVS dataset class for working with event-based vision data.

    Args:
        data_path (str, optional): Path to the data directory. Defaults to "../data".
        train (bool, optional): Whether to load the training set. Defaults to True.
        transform (callable, optional): Optional transform to be applied to the events.
        target_transform (callable, optional): Optional transform to be applied to the target.
        transforms (callable, optional): Optional transform to be applied to both events and target.

    Attributes:
        classes (list): List of classes (letters from 'a' to 'z').
        int_classes (dict): Dictionary mapping classes to integers.
        sensor_size (tuple): Size of the sensor (240x180 pixels with 2 channels).
        dtype (numpy.dtype): Data type for the events (structured array with fields: 't', 'x', 'y', 'p').
        ordering (tuple): Field names in the structured array ('t', 'x', 'y', 'p').

    Notes:
        This dataset class assumes that the data is stored in CSV files where each row contains the file path and label.
        The events are stored in .bin files which can be loaded as numpy arrays.

    Example:
        >>> dataset = ASLDVS(data_path="../my_data", train=True)
        >>> events, target = dataset[0]
    """

    classes = [chr(letter) for letter in range(97, 123)]  # generate alphabet
    int_classes = dict(zip(classes, range(len(classes))))
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names

    def __init__(self,
                 data_path: str, 
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None):
        
        super().__init__(
            data_path, 
            transform=transform, 
            target_transform=target_transform, 
            transforms=transforms)
        self.train = train
        self.sensor_size = (240, 180, 2)

        # Read in file paths and labels from csv files containing metadata
        METADATA = f"{data_path}/train_data.csv" if train else f"{data_path}/test_data.csv"
        df = pd.read_csv(METADATA)

        files = df['file'].to_list()
        for i, f in enumerate(files):
            name = f.split('asl_dataset_v3-003/')[-1]
            name = f"{data_path}/{name}"
            self.data.append(name)

        self.targets = df['label'].to_list()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        # Get the file path and target label
        file_path, target = self.data[index], self.targets[index]

        file_path = file_path.replace("gs://asl-dataset-bucket-v4/", "", 1)
        # Initialize Google Cloud Storage client
        storage_client = storage.Client()

        # Get the bucket and blob objects
        bucket_name = "asl-dataset-bucket-v4"
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)

        # Download the file to a temporary location
        temp_file_path = "/tmp/temp_file.bin"
        blob.download_to_filename(temp_file_path)

        # Load events from the downloaded file
        events = np.load(temp_file_path)

        # Perform transforms on data
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)
        return events, target
           
    def __len__(self):
        return len(self.data)
