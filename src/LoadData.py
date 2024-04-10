import pandas as pd
import numpy as np
import torch


def LoadDataset():

    path = "../data/csv/subject1/a_events.csv"
    
    data = pd.read_csv(path)
    data = data.to_numpy()
    data = torch.tensor(data)

    data = data[:,1:]
    
    return data



