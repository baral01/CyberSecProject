import pandas as pd
import numpy as np
import os

dirpath = "Processed_datasets/TON_IoT/"
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


files = os.listdir(dirpath)
chunkdim = 10000
total_deleted_rows = 0
frames = list()
print("Scanning files and building dataframe...")
for f in files:
    print(f"Opened file {f}")
    with pd.read_csv(dirpath + f, sep=',', dtype=attr_types, chunksize=chunkdim) as reader:
      for chunk in reader:
          # delete rows with wrong values 
          chunk['src_bytes'] = pd.to_numeric(chunk['src_bytes'], errors="coerce")
          chunk = chunk[(~chunk.isnull()).all(axis=1)]
                 
          deleted_rows = chunkdim - chunk.shape[0]
          if deleted_rows > 0:
            print(f"Deleted rows: {deleted_rows}")
          total_deleted_rows = total_deleted_rows + deleted_rows
          frames.append(chunk)

print(f"Total number of deleted rows: {total_deleted_rows}")

dataset = pd.concat(frames)
inp = input("Check memory usage")
cols = dataset.columns.tolist()
cols.remove("uid")
del frames
inp = input("Check memory usage")
print("Saving to csv...")
dataset.to_csv(dirpath + "Network_dataset.csv", sep=',', index=False, index_label=False, columns=cols, header=cols)
print("Done")