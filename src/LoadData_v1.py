import csv
import os
import numpy as np


TRAIN = "D:/UTK/Semester-4/Deep_Learning_New/Project/data/csv/Train"
TEST = "D:/UTK/Semester-4/Deep_Learning_New/Project/data/csv/Test"

LABEL_DICT = {"a":1, "b":2,"c":3, "d":4, "e":5, "f":6, "g":7, "h":8, "i":9, "k":10, "l":11,
             "m":12, "n":13, "o":14, "p":15, "q":16, "r":17, "s":18, "t":19, "u":20, "v":21,
                "w":22, "x":23, "y":24}

# time_step = 100000

# num_samp = 2

train_data_asl = []

test_data_asl = []

for i in os.listdir(TRAIN):
    SUBJECT = f"{TRAIN}/{i}"
    for j in os.listdir(SUBJECT):
        label = j[0]
        label_num = LABEL_DICT[label]
        if j.endswith(".csv"):
            csv_file = f"{SUBJECT}/{j}"
            with open(csv_file,'r') as csv_in:
                csv_reader = csv.reader(csv_in)
                data = [[],[],[],[],[]] # Each entry in the array: x, y, timestamp, polarity, rel. timestamp.
                for row in csv_reader:   
                    if 'timestamp' in row[1]:
                        continue          

                    # Modification for ASL-DVS
                    data[0].append(int(row[2])) # x pixel
                    data[1].append(int(row[3])) # y pixel
                    data[2].append(int(row[1])) # microsec
                    data[3].append(int(row[4])) # polarity

                minClockTime = data[2][0]        
                data[4] = [x - minClockTime for x in data[2]] # relative time

                event = list(zip(data[0],data[1],data[4],data[3]))
                event = np.array(event)
                train_data_asl.append((event, label_num))


            
for i in os.listdir(TEST):
    SUBJECT = f"{TEST}/{i}"
    for j in os.listdir(SUBJECT):
        label = j[0]
        label_num = LABEL_DICT[label]
        if j.endswith(".csv"):
            csv_file = f"{SUBJECT}/{j}"
            with open(csv_file,'r') as csv_in:
                csv_reader = csv.reader(csv_in)
                data = [[],[],[],[],[]] # Each entry in the array: x, y, timestamp, polarity, rel. timestamp.
                for row in csv_reader:   
                    if 'timestamp' in row[1]:
                        continue          

                    # Modification for ASL-DVS
                    data[0].append(int(row[2])) # x pixel
                    data[1].append(int(row[3])) # y pixel
                    data[2].append(int(row[1])) # microsec
                    data[3].append(int(row[4])) # polarity

                minClockTime = data[2][0]        
                data[4] = [x - minClockTime for x in data[2]] # relative time

                event = list(zip(data[0],data[1],data[4],data[3]))
                event = np.array(event)
                test_data_asl.append((event, label_num))





