import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig

from data.ton_iot_dataset import TonIotDataset
import time


# get dataset
filepath = "Processed_datasets/TON_IoT/Network_dataset.csv"
cols = [
    "ts",
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
types = {
    "ts": "int64",  # it is a timestamp
    "src_ip": "string",
    "src_port": "int16",
    "dst_ip": "string",
    "dst_port": "int16",
    "proto": "string",
    "service": "string",
    "duration": "float64",
    "src_bytes": "Int64",
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
    "http_user_agent": "string",  # on description doc, its type is a number
    "http_orig_mime_types": "string",
    "http_resp_mime_types": "string",
    "weird_name": "string",
    "weird_addl": "string",
    "weird_notice": "string",  # boolean in doc
    "label": "Int8",  # only 0 and 1 as numbers: tag normal and attack records
    "type": "string",
}
ton = TonIotDataset(filepath, columns=cols, types=types)
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Loading {filepath}...")
df = ton.load_dataset()
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

###preprocessing

# drop unused label
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Drop 'type' column...")
df.drop(columns=['type'], inplace=True)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

# convert timestamp to datetime and infer frequency
df["ts"] = pd.to_datetime(df["ts"], unit='s')

# separate categorical and numerical features
NUMERICAL_FEATURES = df.select_dtypes(include="number").columns.tolist()
CATEGORICAL_FEATURES = df.select_dtypes(exclude="number").columns.tolist()
print(
    f"Numericals: {len(NUMERICAL_FEATURES)}; Categoricals: {len(CATEGORICAL_FEATURES)}"
)
CATEGORICAL_FEATURES.remove("ts")
NUMERICAL_FEATURES.remove("label")
targets_list = df["label"].unique().tolist()

# split into train and test set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Splitting train and test sets...")
train, test = train_test_split(df, stratify=df["label"], test_size=0.2, random_state=42)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

# prepare model
data_config = DataConfig(
    target=["label"],  # target should always be a list.
    continuous_cols=NUMERICAL_FEATURES,
    categorical_cols=CATEGORICAL_FEATURES,
    date_columns=[("ts", "T", "%d/%m/%Y %H:%M:%S")],
    num_workers=15,
)

trainer_config = TrainerConfig(
    #     auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
    batch_size=128,
    max_epochs=1,
    early_stopping="valid_loss",  # Monitor valid_loss for early stopping
    early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
    early_stopping_patience=5,  # No. of epochs of degradation training will wait before terminating
    checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
    load_best=True,  # After training, load the best checkpoint
)

optimizer_config = OptimizerConfig()

head_config = LinearHeadConfig(
    layers="",  # No additional layer in head, just a mapping layer to output_dim
    dropout=0.1,
    initialization="kaiming",
).__dict__  # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)

model_config = TabTransformerConfig(
    task="classification",
    learning_rate=1e-3,
    head="LinearHead",  # Linear Head
    head_config=head_config,  # Linear Head Config
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
tabular_model.fit(train=train)
tabular_model.evaluate(test)
tabular_model.save_model("Models")
