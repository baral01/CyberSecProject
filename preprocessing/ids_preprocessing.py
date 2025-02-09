import pandas as pd
import numpy as np

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
    """Convert label column into binary format: 0 for benign, 1 for malicious."""
    df.rename(columns={"Label": "label_string"}, inplace=True)
    df['Label'] = np.where(df['label_string'] == "Benign", np.zeros(df['label_string'].shape, dtype='int'), np.ones(df['label_string'].shape, dtype='int'))
    df.drop(columns=['label_string'], inplace=True)
