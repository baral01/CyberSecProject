import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig

from data.cse_ids_dataset import CseIdsDataset
import time


# get dataset
filepath = "Processed_datasets\CSE-CIC-IDS2018-AWS\Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
ids = CseIdsDataset(filepath)
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Loading {filepath}...")
df = ids.load_dataset()
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

###preprocessing

# drop unused label
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Drop 'Flow ID' column...")
df.drop(columns=["Flow ID"], inplace=True)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

# convert timestamp to datetime and infer frequency
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %H:%M:%S")
#freq = pd.infer_freq(df['Timestamp'])
#del df_time

# replace inf values and deal with NaN values by replacing them with 0
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

# separate categorical and numerical features
NUMERICAL_FEATURES = df.select_dtypes(include="number").columns.tolist()
CATEGORICAL_FEATURES = df.select_dtypes(exclude="number").columns.tolist()
print(
    f"Numericals: {len(NUMERICAL_FEATURES)}; Categoricals: {len(CATEGORICAL_FEATURES)}"
)
CATEGORICAL_FEATURES.remove('Timestamp')
CATEGORICAL_FEATURES.remove('Label')
targets_list = df['Label'].unique().tolist()

# split into train and test set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Splitting train and test sets...")
train, test = train_test_split(df, stratify=df["Label"], test_size=0.2, random_state=42)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

# prepare model
data_config = DataConfig(
    target=['Label'],  # target should always be a list.
    continuous_cols=NUMERICAL_FEATURES,
    categorical_cols=CATEGORICAL_FEATURES,
    date_columns=[('Timestamp','T',"%d/%m/%Y %H:%M:%S")],
    num_workers=15
)

trainer_config = TrainerConfig(
    #     auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
    batch_size=256,
    max_epochs=10,
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