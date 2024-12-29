import pandas as pd
import numpy as np

dirpath = "Processed_datasets/CSE-CIC-IDS2018-AWS/"
filepath = "Processed_datasets/CSE-CIC-IDS2018-AWS/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
chunkdim = 10000

def convert_timestamps(df, time_features):
    """Convert Timestamp column into different time features columns listed inside time_features.
    Time features supported: dow, hour, minute, second."""
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %H:%M:%S")
    for f in time_features:        
        if f == "dow":
            #get day of the week
            df['dow'] = df['Timestamp'].dt.day_name()
        elif f == "hour":
            #get hour of the day
            df['hour'] = df['Timestamp'].dt.hour
        elif f == "minute":
            #get minute of the hour
            df['minute'] = df['Timestamp'].dt.minute
        elif f == "second":
            #get second of the minute
            df['second'] = df['Timestamp'].dt.second         
    #drop Timestamp
    df.drop(columns=['Timestamp'], inplace=True)
     
def binarize_label(df):
    df.rename(columns={"Label": "label_string"}, inplace=True)
    df['Label'] = np.where(df['label_string'] == "Benign", np.zeros(df['label_string'].shape, dtype='int'), np.ones(df['label_string'].shape, dtype='int'))
    df.drop(columns=['label_string'], inplace=True)

frames = list()
times_features = ['dow', 'hour', 'minute']
with pd.read_csv(filepath, sep=',', chunksize=chunkdim, index_col=False) as reader:
    for chunk in reader:
        convert_timestamps(chunk, times_features)
        binarize_label(chunk)
        frames.append(chunk)
        
df = pd.concat(frames)
cols = df.columns.tolist()
# for t in times_features:
#     cols.remove(t)
#     cols.insert(cols.index('Timestamp'), t)
# cols.remove('Timestamp')
# cols.remove('label_string')
# df.drop(columns=['Timestamp'], inplace=True)
# df.drop(columns=['label_string'], inplace=True)
del frames

#save new dataset
df.to_csv(dirpath + "Thuesday-20-02-2018_processed.csv", sep=',', index=False, index_label=False, columns=cols)