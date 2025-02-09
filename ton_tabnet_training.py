import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import TabNetModelConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.utils import get_balanced_sampler, get_class_weighted_cross_entropy

import torch
from torchmetrics import ConfusionMatrix, F1Score, AUROC

import data.ton_iot_dataset as ton_iot_dataset
import time
import os


# get dataset
filepath = "Processed_datasets/TON_IoT/Network_dataset.csv"
savepath = "Models/ton/tabnet-" + time.strftime('%Y%m%d-%H%M') + "/"
cols = ton_iot_dataset.COLS
types = ton_iot_dataset.COLS_TYPES

start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Loading {filepath}...")
df = pd.read_csv(filepath, sep=",", usecols=cols, dtype=types)
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
# convert integer types to float, resolve future warning of implicit casting of TabNet
for col in NUMERICAL_FEATURES:
    is_int = pd.api.types.is_integer_dtype(df[col])
    if is_int:
        df[col] = df[col].astype('float64')
NUMERICAL_FEATURES.remove('label')

# split into train and test set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Splitting train and test sets...")
# test_size=0.25 instead of 0.2 in order to fit the train set dimension for the sampler (max 2^24)
train, test = train_test_split(df, stratify=df["label"], test_size=0.25, random_state=42)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")
num_classes = len(set(train["label"].values.ravel()))
num_pos = np.sum(train["label"] == 1)
num_neg = np.sum(train["label"] == 0)
print("Train stats:")
print(f"Number of classes: {num_classes}")
print(f"Number of positive samples: {num_pos}")
print(f"Number of negative samples: {num_neg}")

sampler = get_balanced_sampler(train['label'].values.ravel())
weighted_loss = get_class_weighted_cross_entropy(train["label"].values.ravel(), mu=1.0)

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
    batch_size=256,
    max_epochs=1,
    early_stopping="valid_loss",  # Monitor valid_loss for early stopping
    early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
    early_stopping_patience=5,  # No. of epochs of degradation training will wait before terminating
    checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
    load_best=True,  # After training, load the best checkpoint
    profiler='simple', # https://pytorch-lightning.readthedocs.io/en/1.5.10/advanced/profiler.html
)

optimizer_config = OptimizerConfig()

head_config = LinearHeadConfig(
    layers="",  # No additional layer in head, just a mapping layer to output_dim
    dropout=0.1,
    initialization="kaiming",
).__dict__  # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)

model_config = TabNetModelConfig(
    task="classification",
    learning_rate=1e-3,
    head="LinearHead",  # Linear Head
    head_config=head_config,  # Linear Head Config
    metrics=['accuracy', 'f1_score', 'precision', 'recall', 'auroc'], #found in Lib/site-packages/torchmetrics/functional/__init__.py
    metrics_prob_input=[False, False, False, False, True],
    metrics_params=[{'task': 'binary', 'num_classes': num_classes}, {'task': 'binary', 'num_classes': num_classes}, {'task': 'binary', 'num_classes': num_classes}, {'task': 'binary', 'num_classes': num_classes}, {}],
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
# training_type = ["simple", "sampler", "weighted"]
training_type = "weighted"

time_str = time.strftime("%R")
print(f"<{time_str}> Model initialized.")
start_training_time = time.time()
if training_type == "simple":
    tabular_model.fit(train=train, validation=test)
elif training_type == "sampler":
    tabular_model.fit(train=train, validation=test, sampler=sampler)
elif training_type == "weighted":
    tabular_model.fit(train=train, validation=test, loss=weighted_loss)
end_training_time = time.time()
training_time = end_training_time - start_training_time
time_str = time.strftime("%R")
print(f"<{time_str}> Model training completed. Elapsed {training_time}s.")
# training can generate user warnings along the lines of "no positive/negative samples in target"
# that is caused by some metrics being calculated at the end of each batch

time_str = time.strftime("%R")
print(f"<{time_str}>Saving model...")
os.makedirs(savepath)
tabular_model.save_model(savepath)
time_str = time.strftime("%R")
print(f"<{time_str}>Model saved.")

num_classes = len(set(train["label"].values.ravel()))
num_pos = np.sum(train["label"] == 1)
num_neg = np.sum(train["label"] == 0)
print("Train stats:")
print(f"Number of classes: {num_classes}")
print(f"Number of positive samples: {num_pos}")
print(f"Number of negative samples: {num_neg}")

time_str = time.strftime("%R")
print(f"<{time_str}>Evaluating model...")
start_evaluation_time = time.time()
scores = tabular_model.evaluate(test)
end_evaluation_time = time.time()
evaluation_time = end_evaluation_time - start_evaluation_time
time_str = time.strftime("%R")
print(f"<{time_str}>Model evaluation completed. Elapsed {evaluation_time}s.")

time_str = time.strftime("%R")
print(f"<{time_str}>Test predictions (obtain average inference time)...")
test_rows = test.shape[0]
start_predictions_time = time.time()
# predictions columns: [benign_probability, attack_probability, predicted_label]
predictions = tabular_model.predict(test=test)
end_predictions_time = time.time()
predictions_time = end_predictions_time - start_predictions_time
time_str = time.strftime("%R")
print(f"<{time_str}>Predictions completed. Elapsed {predictions_time}s.")
average_inference_time = predictions_time / test_rows
print(f"Average inference time: {average_inference_time}s.")

print("---Other metrics---")
probs = torch.tensor((predictions.iloc[:, 1]).values)
preds = torch.tensor((predictions.iloc[:, 2]).values)
test_target = torch.tensor(test['label'].values)
confmat = ConfusionMatrix(task='binary', num_classes=num_classes)
confmat_res = confmat(preds, test_target)
print("Confusion Matrix:")
print(f"True Negatives: {confmat_res[0, 0].item()}")
print(f"False Positives: {confmat_res[0, 1].item()}")
print(f"False Negatives: {confmat_res[1, 0].item()}")
print(f"True Positives: {confmat_res[1, 1].item()}")
(scores[0]).update({
    "TN": confmat_res[0, 0].item(),
    "FP": confmat_res[0, 1].item(),
    "FN": confmat_res[1, 0].item(),
    "TP": confmat_res[1, 1].item(),
    "avg_inf_time": average_inference_time,
    "training_time": training_time,
})

f1 = F1Score(task='binary', num_classes=num_classes)
f1_res = f1(preds, test_target)
print(f"F1_Score: {f1_res.item()}")
auroc = AUROC(task="binary")
auroc_res = auroc(probs, test_target)
print(f"AUROC_Score: {auroc_res.item()}")

# Save scores to a .txt file
scores_txt_path = savepath + "scores.txt"
with open(scores_txt_path, "w") as f:
    for metric, score in (scores[0]).items():
        f.write(f"{metric}: {score}\n")

# Save scores to a .csv file
scores_csv_path = savepath + "scores.csv"
scores_df = pd.DataFrame(list((scores[0]).items()), columns=["Metric", "Value"])
scores_df.to_csv(scores_csv_path, index=False)