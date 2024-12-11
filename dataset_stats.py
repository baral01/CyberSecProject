import pandas as pd
import numpy as np
import os

dirpath = "Processed_datasets/TON_IoT/"
filepath = "Processed_datasets/TON_IoT/Network_dataset_1.csv"
attr_types = {
              "ts": "int64", # it is a timestamp
              "src_ip": "string",
              "src_port": "int16",
              "dst_ip": "string",
              "dst_port": "int16",
              "proto": "string",
              "service": "string",
              "duration": "float64",
              "src_bytes": "string", #problematic column for parsing returning a "0.0.0.0" could not be parsed. it's a number type
              "dst_bytes": "Int64",
              "conn_state": "string",
              "missed_bytes": "Int64",
              "src_pkts": "Int64",
              "src_ip_bytes": "Int64",
              "dst_pkts": "Int64",
              "dst_ip_bytes": "Int64",
              "dns_query": "string",
              "dns_qclass": "Int32",
              "dns_qtype": "Int32",
              "dns_rcode": "Int32",
              "dns_AA": "string", #boolean in doc
              "dns_RD": "string", #boolean in doc
              "dns_RA": "string", #boolean in doc
              "dns_rejected": "string", #boolean in doc
              "ssl_version": "string",
              "ssl_cipher": "string",
              "ssl_resumed": "string", #boolean in doc
              "ssl_established": "string",
              "ssl_subject": "string",
              "ssl_issuer": "string",
              "http_trans_depth": "string", # on description doc, its type is a number
              "http_method": "string",
              "http_uri": "string",
              "http_referrer": "string", # not present on description doc
              "http_version": "string",
              "http_request_body_len": "Int64",
              "http_response_body_len": "Int64",
              "http_status_code": "Int16",
              "http_user_agent": "string", # on description doc, its type is a number
              "http_orig_mime_types": "string",
              "http_resp_mime_types": "string",
              "weird_name": "string",
              "weird_addl": "string",
              "weird_notice": "string", #boolean in doc
              "label": "Int8", # only 0 and 1 as numbers: tag normal and attack records
              "type": "string"
            }

headers = pd.read_csv(filepath, sep=',', index_col=False, nrows=0,).columns.tolist()
print(f"Number of fields: {len(headers)}")
print(f"Fields: {headers}")

df = pd.read_csv(filepath, sep=',', )
columns_types = dict(df.dtypes)
print(columns_types)

nrows = df.shape[0]
print(f"Number of rows: {nrows}")

### count the number of nunique and non-null values for each column
# get files
files = os.listdir(dirpath)

#dictionary {"column_name": (nunique, non-null)}
values_stats = dict()

for column in headers:
    frames = list()
    
    for f in files:
        partial_df = pd.read_csv(dirpath + f, sep=',', dtype=attr_types, usecols=[column])
        frames.append(partial_df)
        
    dataset = pd.concat(frames)
    values_stats.update({column: (dataset[column].nunique(), dataset[column].count())})
    del frames

print(values_stats)
