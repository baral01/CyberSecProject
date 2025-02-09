import numpy as np
import pandas as pd
import os
import time

dirpath = "Processed_datasets/CSE-CIC-IDS2018-AWS/cleaned/"
filepath = "Processed_datasets/CSE-CIC-IDS2018-AWS/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"

files = os.listdir(dirpath)
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Loading dataset...")
frames = list()
for f in files:
    print(f"Loaded file {f}")
    partial_df = pd.read_csv(dirpath + f, sep=',', usecols=['Label'])
    frames.append(partial_df)
print("Combining frames...")
df = pd.concat(frames)
del frames
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

unique_values = df['Label'].unique().tolist()

df.rename(columns={"Label": "label_string"}, inplace=True)
df['Label'] = np.where(df['label_string'] == "Benign", np.zeros(df['label_string'].shape, dtype='int'), np.ones(df['label_string'].shape, dtype='int'))
df.drop(columns=['label_string'], inplace=True)

normal_count = (df['Label'] == 0).sum()
attack_count = (df['Label'] == 1).sum()

print(f"Unique label values: {unique_values}")
print(f"Normal traffic count: {normal_count}")
print(f"Attack traffic count: {attack_count}")
