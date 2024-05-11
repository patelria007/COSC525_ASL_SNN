"""
Convert AEDAT4 files to CSV format.

This script converts AEDAT4 files to CSV format. It uses the 
`dv_processing` library for handling AEDAT4 files and the 
`argparse` module for parsing command-line arguments.

Usage:
    python script.py --file <path_to_aedat4_file> --dir <output_directory>

Arguments:
    --file (str): Path to the AEDAT4 file to be converted.
    --dir (str): Path to the directory where the CSV file will be saved.
"""


import dv_processing as dv
import argparse
import pathlib
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Save aedat4 data to csv.')

parser.add_argument('--file', type=str, required=True, help='Path to an AEDAT4 file')
parser.add_argument('--dir',  type=str, required=True, help='Path to save CSV')
args = parser.parse_args()

file = pathlib.Path(args.file)
file_parent = file.parent
file_stem = file.stem
export_csv = args.dir

# Open the recording file
recording = dv.io.MonoCameraRecording(args.file)

if recording.isEventStreamAvailable():
    events_path = f"{export_csv}/{file_stem}_events.csv"
    print("Events will be saved under %s" % events_path)

    print("Saving Events...")
    events_packets = []
    while True:
        events = recording.getNextEventBatch()
        if events is None:
            break
        events_packets.append(pd.DataFrame(events.numpy()))
    print("Done reading events, saving into CSV...")
    events_pandas = pd.concat(events_packets)
    events_pandas.to_csv(events_path)
    print("Saved %d events." % len(events_pandas))
