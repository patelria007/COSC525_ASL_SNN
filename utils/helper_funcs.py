import pandas as pd
import matplotlib.pyplot as plt
import dv_processing as dv
from PIL import Image
import numpy as np

def getRecordingDuration(record):
    """
    Calculate the duration of a recording.

    Args:
    record (dv.io.BaseRecording): The recording object.

    Returns:
    float: The duration of the recording in seconds.
    """
    if record.isEventStreamAvailable():
        start, end = record.getTimeRange()
        duration = end - start 
    
    return duration # divide by 1e6 to convert to seconds

def getSampleDuration(sample):
    """
    Calculate the duration of a sample.

    Args:
    sample (dv.events.EventPacket): The event packet representing a sample.

    Returns:
    float: The duration of the sample in microseconds.
    """
    start, end = sample.timestamps()[0], sample.timestamps()[-1]
    duration = end - start
    return duration

def getNumEvents(sample):
    """
    Get the number of events in a sample.

    Args:
    sample (dv.events.EventPacket): The event packet representing a sample.

    Returns:
    int: The number of events in the sample.
    """
    num_events = sample.size()
    return num_events

def split_recording(record, batch_time, PRINT=False):
    """
    Split a recording into batches.

    Args:
    record (dv.io.BaseRecording): The recording object.
    batch_time (float): The duration of each batch in microseconds.
    PRINT (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
    tuple: A tuple containing metadata about the recording.
    """
    record_duration = getRecordingDuration(record) # Total duration of recording
    resolution = record.getEventResolution() # Resolution of camera
    t0, t1 = record.getTimeRange() # Start & end timestamps of recording
    num_batches = record_duration // batch_time # Number of batches in recording (floored)

    if PRINT:
        print(f"Event resolution: {resolution}")
        print(f"Recording duration: {record_duration / 1e6} secs")
        print("Start timestamp:", t0)
        print("End timestamp:", t1)
    
    return (resolution, batch_time, num_batches, t0, t1)  # Return metadata

def get_batches(record, metadata):
    """
    Get batches from a recording.

    Args:
    record (dv.io.BaseRecording): The recording object.
    metadata (tuple): Metadata about the recording.

    Returns:
    pd.DataFrame: A DataFrame containing information about each batch.
    """
    _, batch_time, num_batches, t0, _ = metadata # Unpacks metadata
    cols = ['batch', 't0', 't1', 'duration', 'num_events']
    batches = pd.DataFrame(columns=cols)
    start = t0
    for i in range(num_batches):
        start, end = start, start + batch_time
        batch = record.getEventsTimeRange(start, end)
        batch_duration = getSampleDuration(batch)
        batch_numEvents = getNumEvents(batch)

        data = pd.DataFrame({'batch': [i], 
                             't0': [start], 
                             't1': [end], 
                             'duration': [batch_duration], 
                             'num_events': [batch_numEvents]})
        batches = pd.concat([batches, data])
        start += batch_time

    return batches 

def get_batch_indices(df, batches, N, PRINT=False):
    """
    Get indices of events in a batch.

    Args:
    df (pd.DataFrame): The DataFrame containing all events.
    batches (pd.DataFrame): The DataFrame containing batch information.
    N (int): The index of the batch.
    PRINT (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
    pd.DataFrame: A DataFrame containing the events in the specified batch.
    """
    num_events = batches['num_events'].iloc[N] # Gets Batch N 
    
    idx0 = batches['t0'].iloc[N]
    idx1 = batches['t1'].iloc[N]
    batch = df[(df['timestamp'] > idx0) & (df['timestamp'] < idx1)]

    if PRINT:
        print(f"Number of events in Batch {N}: {num_events}")
    batch.reset_index(drop=True, inplace=True)
    return batch

def sample_2_image(sampleN, metadata):
    """
    Convert a sample to an image.

    Args:
    sampleN (pd.DataFrame): The DataFrame representing a sample.
    metadata (tuple): Metadata about the recording.

    Returns:
    np.ndarray: The image representation of the sample.
    """
    res, _, _, _, _ = metadata
    img = np.full(res, 1)
    for i in range(sampleN.shape[0]):
        x, y = sampleN['x'].iloc[i], sampleN['y'].iloc[i]
        img[x,y] = 0
    return img.T

def get_sample_freq_from_batch(batch, metadata, IMGS=False, PRINT=False):
    """
    Get the sampling frequency from a batch.

    Args:
    batch (pd.DataFrame): The DataFrame containing events in a batch.
    metadata (tuple): Metadata about the recording.
    IMGS (bool, optional): Whether to generate images. Defaults to False.
    PRINT (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
    tuple: A tuple containing the sampling frequencies and images.
    """
    t0s = list(batch[batch['Unnamed: 0'] == 0].index)
    size = len(t0s)

    # Get # of events in each sample (gets sampling frequency)
    sample_freq = []
    imgs = []
    for i, t0 in enumerate(t0s):
        sample = batch.iloc[t0:t0s[i+1]] if i+1 < size else batch.iloc[t0:]
        sample_freq.append(sample.size)
        
        # Gets first image in first sample
        if IMGS:
            img = sample_2_image(sample, metadata)
            imgs.append(img)
            IMGS = False
            if PRINT: 
                print("# of events in first sample:", sample.size)
    
    if PRINT:
        print("# of samples in batch:", len(t0s))
        print("# of samples in batch (frequency check):", len(sample_freq))

    return np.array(sample_freq), np.array(imgs)