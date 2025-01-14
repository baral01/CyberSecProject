import pandas as pd
import os


class TonIotDataset:

    def __init__(self, path, columns=None, types=None):
        self.path = path
        self.dirpath = os.path.dirname(path)

        if columns is None:
            self.__columns = [
                "dow",
                "hour",
                "minute",
                "src_ip",
                "src_port",
                "dst_ip",
                "dst_port",
                "proto",
                "service",
                "duration",
                "src_bytes",
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
                "dns_AA",
                "dns_RD",
                "dns_RA",
                "dns_rejected",
                "ssl_version",
                "ssl_cipher",
                "ssl_resumed",
                "ssl_established",
                "ssl_subject",
                "ssl_issuer",
                "http_trans_depth",
                "http_method",
                "http_uri",
                "http_referrer",
                "http_version",
                "http_request_body_len",
                "http_response_body_len",
                "http_status_code",
                "http_user_agent",
                "http_orig_mime_types",
                "http_resp_mime_types",
                "weird_name",
                "weird_addl",
                "weird_notice",
                "label",
                "type",
            ]
        else:
            self.__columns = columns

        if types is None:
            self.__types = {
                "dow": "string",
                "hour": "int8",
                "minute": "int8",
                "src_ip": "string",
                "src_port": "int32",
                "dst_ip": "string",
                "dst_port": "int32",
                "proto": "string",
                "service": "string",
                "duration": "float64",
                "src_bytes": "int64",
                "dst_bytes": "int64",
                "conn_state": "string",
                "missed_bytes": "int64",
                "src_pkts": "int64",
                "src_ip_bytes": "int64",
                "dst_pkts": "int64",
                "dst_ip_bytes": "int64",
                "dns_query": "string",
                "dns_qclass": "int32",
                "dns_qtype": "int32",
                "dns_rcode": "int32",
                "dns_AA": "string",  # boolean in doc
                "dns_RD": "string",  # boolean in doc
                "dns_RA": "string",  # boolean in doc
                "dns_rejected": "string",  # boolean in doc
                "ssl_version": "string",
                "ssl_cipher": "string",
                "ssl_resumed": "string",  # boolean in doc
                "ssl_established": "string",
                "ssl_subject": "string",
                "ssl_issuer": "string",
                "http_trans_depth": "string",  # on description doc, its type is a number
                "http_method": "string",
                "http_uri": "string",
                "http_referrer": "string",  # not present on description doc
                "http_version": "string",
                "http_request_body_len": "Int64",
                "http_response_body_len": "Int64",
                "http_status_code": "Int16",
                "http_user_agent": "string",  # on description doc this field's type is a number
                "http_orig_mime_types": "string",
                "http_resp_mime_types": "string",
                "weird_name": "string",
                "weird_addl": "string",
                "weird_notice": "string",  # boolean in doc
                "label": "Int8",  # only 0 and 1 as numbers: tag normal and attack records
                "type": "string",
            }
        else:
            self.__types = types

    def load_dataset(self, cols=None):
        if cols is None:
            cols = self.__columns
        self.cols_in_use = cols
        df = pd.read_csv(self.path, sep=",", usecols=cols, dtype=self.__types)
        return df

    def save_dataset(self, dataframe, filename, dirpath=None):
        if dirpath is None:
            dirpath = self.dirpath
        save_path = dirpath + filename
        dataframe.to_csv(save_path, sep=",", index=False, index_label=False)
