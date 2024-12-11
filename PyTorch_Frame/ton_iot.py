import pandas as pd

import torch_frame


class NetworkDataTon(torch_frame.data.Dataset):
    r"""The `TON_IoT 
    <https://research.unsw.edu.au/projects/toniot-datasets>`_
    datasets from UNSW Canberra at ADFA. They include heterogeneous
    data sources collected from Telemetry datasets of IoT and IIoT sensors,
    Operating systems datasets of Windows 7 and 10 as well as
    Ubuntu 14 and 18 TLS and Network traffic datasets.

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 20 10
        :header-rows: 1

        * - #rows
          - #cols (numerical)
          - #cols (categorical)
          - #classes
          - Task
          - Missing value ratio
        * - 32,561
          - 4
          - 8
          - 2
          - binary_classification
          - 0.0%
    """

    def __init__(self, root: str, dataframe=None):
        path = root + "Network_dataset_1.csv"
        
        names = [
            "ts", # it is a timestamp
            "src_ip",
            "src_port",
            "dst_ip",
            "dst_port",
            "proto",
            "service",
            "duration",
            "src_bytes", #problematic column for parsing returning a "0.0.0.0" could not be parsed. it's a number type
            "dst_bytes",
            "conn_state",
            "missed_bytes",
            "src_pkts",
            "src_ip_bytes",
            "dst_pkts",
            "dst_ip_bytes",
            "dns_query",
            "dns_qclass",
            "dns_qtype",
            "dns_rcode",
            "dns_AA", #boolean in doc
            "dns_RD", #boolean in doc
            "dns_RA", #boolean in doc
            "dns_rejected", #boolean in doc
            "ssl_version",
            "ssl_cipher",
            "ssl_resumed", #boolean in doc
            "ssl_established",
            "ssl_subject",
            "ssl_issuer",
            "http_trans_depth", # on description doc, its type is a number
            "http_method",
            "http_uri",
            "http_referrer", # not present on description doc
            "http_version",
            "http_request_body_len",
            "http_response_body_len",
            "http_status_code",
            "http_user_agent", # on description doc, its type is a number
            "http_orig_mime_types",
            "http_resp_mime_types",
            "weird_name",
            "weird_addl",
            "weird_notice", #boolean in doc
            #"label", # only 0 and 1 as numbers: tag normal and attack records, skipped because redundant
            "type",
        ]
        
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
        
        if dataframe is None:
            df = pd.read_csv(path, usecols=names, dtype=attr_types)
        else:
            df = dataframe

        col_to_stype = {
            "ts": torch_frame.timestamp, # it is a timestamp
            "src_ip": torch_frame.categorical,
            "src_port": torch_frame.categorical,
            "dst_ip": torch_frame.categorical,
            "dst_port": torch_frame.categorical,
            "proto": torch_frame.categorical,
            "service": torch_frame.categorical,
            "duration": torch_frame.numerical,
            "src_bytes": torch_frame.categorical, #problematic column for parsing returning a "0.0.0.0" could not be parsed. it's a number type
            "dst_bytes": torch_frame.numerical,
            "conn_state": torch_frame.categorical,
            "missed_bytes": torch_frame.numerical,
            "src_pkts": torch_frame.numerical,
            "src_ip_bytes": torch_frame.numerical,
            "dst_pkts": torch_frame.numerical,
            "dst_ip_bytes": torch_frame.numerical,
            "dns_query": torch_frame.categorical,
            "dns_qclass": torch_frame.categorical,
            "dns_qtype": torch_frame.categorical,
            "dns_rcode": torch_frame.categorical,
            "dns_AA": torch_frame.categorical, #boolean in doc
            "dns_RD": torch_frame.categorical, #boolean in doc
            "dns_RA": torch_frame.categorical, #boolean in doc
            "dns_rejected": torch_frame.categorical, #boolean in doc
            "ssl_version": torch_frame.categorical,
            "ssl_cipher": torch_frame.categorical,
            "ssl_resumed": torch_frame.categorical, #boolean in doc
            "ssl_established": torch_frame.categorical,
            "ssl_subject": torch_frame.categorical,
            "ssl_issuer": torch_frame.categorical,
            "http_trans_depth": torch_frame.categorical, # on description doc, its type is a number
            "http_method": torch_frame.categorical,
            "http_uri": torch_frame.categorical,
            "http_referrer": torch_frame.categorical, # not present on description doc
            "http_version": torch_frame.categorical,
            "http_request_body_len": torch_frame.numerical,
            "http_response_body_len": torch_frame.numerical,
            "http_status_code": torch_frame.categorical,
            "http_user_agent": torch_frame.categorical, # on description doc, its type is a number
            "http_orig_mime_types": torch_frame.categorical,
            "http_resp_mime_types": torch_frame.categorical,
            "weird_name": torch_frame.categorical,
            "weird_addl": torch_frame.categorical,
            "weird_notice": torch_frame.categorical, #boolean in doc
            #"label", # only 0 and 1 as numbers: tag normal and attack records, skipped because redundant
            "type": torch_frame.categorical,
        }

        super().__init__(df, col_to_stype, target_col='type')
