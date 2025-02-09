import pandas as pd

def convert_timestamps(df, time_features):
    """Convert ts (timestamps) column into different time features columns listed inside time_features.
    Time features supported: dow, hour, minute, second."""
    
    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    for f in time_features:        
        if f == "dow":
            #get day of the week
            df['dow'] = df['ts'].dt.day_name()
        elif f == "hour":
            #get hour of the day
            df['hour'] = df['ts'].dt.hour
        elif f == "minute":
            #get minute of the hour
            df['minute'] = df['ts'].dt.minute
        elif f == "second":
            #get second of the minute
            df['second'] = df['ts'].dt.second         
    #drop ts
    df.drop(columns=['ts'], inplace=True)
