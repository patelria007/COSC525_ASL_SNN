from tonic.dataset import Dataset
from typing import Any, Callable, Optional, Tuple
import pandas as pd
import numpy as np

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
            name = f.split('data/')[-1]
            name = f"{data_path}/{name}"
            self.data.append(name)



        # self.data = df['file'].to_list() # Contains file paths
        self.targets = df['label'].to_list()

        # PATH = f"{data_path}/train" if train else f"{data_path}/test"
        # for file in sorted(os.listdir(PATH)):
        #     if file.endswith("bin"):
        #         FILE_PATH = f"{PATH}/{file}"
        #         label = file.split('_')[0]
        #         self.data.append(FILE_PATH)
        #         self.targets.append(self.int_classes[label])


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            (events, target) where target is index of the target class.
        """
        # events, target = scio.loadmat(self.data[index]), self.targets[index]
        # events = make_structured_array(
        #     events["ts"],
        #     events["x"],
        #     self.sensor_size[1] - 1 - events["y"],
        #     events["pol"],
        #     dtype=self.dtype,
        # )

        # Get events and target from .bin file
        FILE, target = self.data[index], self.targets[index]
        with open(FILE, 'rb') as f:
            events = np.load(f)

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


    # def _check_exists(self):
    #     return (
    #         self._is_file_present()
    #         and self._folder_contains_at_least_n_files_of_type(100800, ".mat")
    #     )