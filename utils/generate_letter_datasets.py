"""
Generate datasets in CSV format for each letter in the alphabet.

This script generates training and testing datasets in CSV format 
for each letter in the alphabet for each subject. It uses the 
`generate_datasets_csv` function from `helper_funcs.py` to 
process recordings and create the datasets. The duration of 
each batch is set to 3 seconds (3e6 microseconds).

Usage:
    python gen_train_test_files.py

"""

from helper_funcs import *

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 
           't', 'u', 'v', 'w', 'x', 'y', 'z']


batch_time = int(3e6) # Batches of 3 seconds of recording

for l in letters:
    generate_datasets_csv(l, batch_time)
