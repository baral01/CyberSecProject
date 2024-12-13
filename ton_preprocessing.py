import pandas as pd

dirpath = "Processed_datasets/TON_IoT/"
filepath = "Processed_datasets/TON_IoT/Network_dataset.csv"
chunkdim = 10000
chunks = list()

#extract infos (day of the week, hour and minute) from timestamp
with pd.read_csv(filepath, sep=',', chunksize=chunkdim, usecols=["ts"]) as reader:
    for chunk in reader:
        chunk['ts'] = pd.to_datetime(chunk['ts'], unit='s')
        #get day of the week
        chunk['dow'] = chunk['ts'].dt.day_name()
        #get hour of the day
        chunk['hour'] = chunk['ts'].dt.hour
        #get minute of the day
        chunk['minute'] = chunk['ts'].dt.minute
        chunk.drop(columns=['ts'], inplace=True)
        chunks.append(chunk)

#concatenate chunks        
time_data = pd.concat(chunks)
print(f"Unique values: {time_data['dow'].nunique()}, {time_data['hour'].nunique()}, {time_data['minute'].nunique()}")
#free memory
del chunks

#optional: save time dataframe to disk
#time_data.to_csv(dirpath + "Network_dataset_time_data.csv", sep=',', index=False, index_label=False, columns=['dow', 'hour', 'minute'], header=['dow', 'hour', 'minute'])

###assemble new whole dataset
#get columns needed from original dataset
headers = pd.read_csv(filepath, sep=',', index_col=False, nrows=0,).columns.tolist()
headers.remove('ts')
original_df = pd.read_csv(filepath, sep=',', usecols=headers)
#concatenate on the vertical axis
data = [time_data, original_df]
df = pd.concat(data, axis=1)
#free memory
del data
#save new dataset
df.to_csv(dirpath + "Network_dataset_ts_extracted.csv", sep=',', index=False, index_label=False)