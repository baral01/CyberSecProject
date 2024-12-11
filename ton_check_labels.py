import pandas as pd
import numpy as np
import os

filepath = "Processed_datasets/TON_IoT/Network_dataset.csv"
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


#check if there are instances of records with label=0 and type!='normal'
collection = list()
with pd.read_csv(filepath, sep=',', dtype=attr_types, chunksize=128, usecols=['label', 'type']) as reader:
    for chunk in reader:
        filtered_rows = chunk.query("label == 0 and type != 'normal'")
        if not filtered_rows.empty:
            print("Partial result:")
            print(filtered_rows)
        collection.append(filtered_rows)

result = pd.concat(collection)
result.to_csv("Processed_datasets/TON_IoT/test0.csv", sep=',', index=False)
#result was an empty data frame (good)

#check if there are instances of records with label=1 and type='normal'
collection = list()
with pd.read_csv(filepath, sep=',', dtype=attr_types, chunksize=128, usecols=['label', 'type']) as reader:
    for chunk in reader:
        filtered_rows = chunk.query("label == 1 and type == 'normal'")
        if not filtered_rows.empty:
            print("Partial result:")
            print(filtered_rows)
        collection.append(filtered_rows)

result = pd.concat(collection)
result.to_csv("Processed_datasets/TON_IoT/test1.csv", sep=',', index=False)
#result was an empty data frame (good)
