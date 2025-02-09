import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import TabTransformerConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig

import torch
from torchmetrics import ConfusionMatrix, F1Score, AUROC

import preprocessing.ids_preprocessing as ids_preprocessing
import time
import os


dirpath = "Processed_datasets/CSE-CIC-IDS2018-AWS/cleaned/"
proc_cols = [
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",
    "Fwd Pkt Len Std",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean",
    "Bwd Pkt Len Std",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Tot",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Len",
    "Bwd Header Len",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Min",
    "Pkt Len Max",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "CWE Flag Count",
    "ECE Flag Cnt",
    "Down/Up Ratio",
    "Pkt Size Avg",
    "Fwd Seg Size Avg",
    "Bwd Seg Size Avg",
    "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Pkts",
    "Subflow Bwd Byts",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Label",
]
proc_types = {
    "Dst Port": "int64",
    "Protocol": "int64",
    "Timestamp": "string",
    "Flow Duration": "float64",
    "Tot Fwd Pkts": "float64",
    "Tot Bwd Pkts": "float64",
    "TotLen Fwd Pkts": "float64",
    "TotLen Bwd Pkts": "float64",
    "Fwd Pkt Len Max": "float64",
    "Fwd Pkt Len Min": "float64",
    "Fwd Pkt Len Mean": "float64",
    "Fwd Pkt Len Std": "float64",
    "Bwd Pkt Len Max": "float64",
    "Bwd Pkt Len Min": "float64",
    "Bwd Pkt Len Mean": "float64",
    "Bwd Pkt Len Std": "float64",
    "Flow Byts/s": "float64",
    "Flow Pkts/s": "float64",
    "Flow IAT Mean": "float64",
    "Flow IAT Std": "float64",
    "Flow IAT Max": "float64",
    "Flow IAT Min": "float64",
    "Fwd IAT Tot": "float64",
    "Fwd IAT Mean": "float64",
    "Fwd IAT Std": "float64",
    "Fwd IAT Max": "float64",
    "Fwd IAT Min": "float64",
    "Bwd IAT Tot": "float64",
    "Bwd IAT Mean": "float64",
    "Bwd IAT Std": "float64",
    "Bwd IAT Max": "float64",
    "Bwd IAT Min": "float64",
    "Fwd PSH Flags": "float64",
    "Bwd PSH Flags": "float64",
    "Fwd URG Flags": "float64",
    "Bwd URG Flags": "float64",
    "Fwd Header Len": "float64",
    "Bwd Header Len": "float64",
    "Fwd Pkts/s": "float64",
    "Bwd Pkts/s": "float64",
    "Pkt Len Min": "float64",
    "Pkt Len Max": "float64",
    "Pkt Len Mean": "float64",
    "Pkt Len Std": "float64",
    "Pkt Len Var": "float64",
    "FIN Flag Cnt": "float64",
    "SYN Flag Cnt": "float64",
    "RST Flag Cnt": "float64",
    "PSH Flag Cnt": "float64",
    "ACK Flag Cnt": "float64",
    "URG Flag Cnt": "float64",
    "CWE Flag Count": "float64",
    "ECE Flag Cnt": "float64",
    "Down/Up Ratio": "float64",
    "Pkt Size Avg": "float64",
    "Fwd Seg Size Avg": "float64",
    "Bwd Seg Size Avg": "float64",
    "Fwd Byts/b Avg": "float64",
    "Fwd Pkts/b Avg": "float64",
    "Fwd Blk Rate Avg": "float64",
    "Bwd Byts/b Avg": "float64",
    "Bwd Pkts/b Avg": "float64",
    "Bwd Blk Rate Avg": "float64",
    "Subflow Fwd Pkts": "float64",
    "Subflow Fwd Byts": "float64",
    "Subflow Bwd Pkts": "float64",
    "Subflow Bwd Byts": "float64",
    "Init Fwd Win Byts": "float64",
    "Init Bwd Win Byts": "float64",
    "Fwd Act Data Pkts": "float64",
    "Fwd Seg Size Min": "float64",
    "Active Mean": "float64",
    "Active Std": "float64",
    "Active Max": "float64",
    "Active Min": "float64",
    "Idle Mean": "float64",
    "Idle Std": "float64",
    "Idle Max": "float64",
    "Idle Min": "float64",
    "Label": "string",
}
files = os.listdir(dirpath)
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Loading dataset...")
frames = list()
for f in files:
    print(f"Loaded file {f}")
    partial_df = pd.read_csv(dirpath + f, sep=',', usecols=proc_cols, dtype=proc_types)
    frames.append(partial_df)
print("Combining frames...")
df = pd.concat(frames)
del frames
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

###preprocessing

# binarize labels for benign/malicious traffic
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Processing Label column for binary classifier...")
ids_preprocessing.binarize_label(df)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

# convert timestamp to datetime and infer frequency
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Converting timestamp to datetime data...")
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %H:%M:%S")
#freq = pd.infer_freq(df['Timestamp'])
#del df_time
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

# replace inf values and deal with NaN values by replacing them with 0
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Replacing infinite and NaN values...")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

# separate categorical and numerical features
NUMERICAL_FEATURES = df.select_dtypes(include="number").columns.tolist()
CATEGORICAL_FEATURES = df.select_dtypes(exclude="number").columns.tolist()
print(
    f"Numericals: {len(NUMERICAL_FEATURES)}; Categoricals: {len(CATEGORICAL_FEATURES)}"
)
CATEGORICAL_FEATURES.remove('Timestamp')
NUMERICAL_FEATURES.remove('Label')

# convert integer types to float, resolve future warning of implicit casting of TabTransformer
for col in NUMERICAL_FEATURES:
    is_int = pd.api.types.is_integer_dtype(df[col])
    if is_int:
        df[col] = df[col].astype('float64')

# split into train and test set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Splitting train and test sets...")
train, test = train_test_split(df, stratify=df["Label"], test_size=0.2, random_state=42)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")
num_classes = len(set(train["Label"].values.ravel()))

# Load the saved model
savepath = "Models/ids2018/tabtransformer-20250117-0034/"
tabular_model = TabularModel.load_model(savepath)

time_str = time.strftime("%R")
print(f"<{time_str}>Evaluating model...")
start_evaluation_time = time.time()
scores = tabular_model.evaluate(test)
end_evaluation_time = time.time()
evaluation_time = end_evaluation_time - start_evaluation_time
time_str = time.strftime("%R")
print(f"<{time_str}>Model evaluation completed. Elapsed {evaluation_time}s.")
print("Scores:")
print(scores)

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
print("Predictions:")
print(predictions.head())  # Check the first few rows of predictions

# Extract probabilities and predicted labels
probs = torch.tensor((predictions.iloc[:, 1]).values)
preds = torch.tensor((predictions.iloc[:, 2]).values)
print("Probs:")
print(probs[:10])  # Check the first few probability values
print("Preds:")
print(preds[:10])  # Check the first few predicted labels

# Ensure test_target is correct
test_target = torch.tensor(test['Label'].values)
print("Test Target:")
print(test_target[:10])  # Check the first few true labels

# Compute confusion matrix
confmat = ConfusionMatrix(task='binary', num_classes=num_classes)
confmat_res = confmat(preds, test_target)
print(f"True Negatives: {confmat_res[0, 0]}")
print(f"False Positives: {confmat_res[0, 1]}")
print(f"False Negatives: {confmat_res[1, 0]}")
print(f"True Positives: {confmat_res[1, 1]}")
print(scores)
print(scores[0])
(scores[0]).update({
    "TN": confmat_res[0, 0],
    "FP": confmat_res[0, 1],
    "FN": confmat_res[1, 0],
    "TP": confmat_res[1, 1],
    "avg_inf_time": average_inference_time,
})
print(scores)
print(scores[0])

# Compute F1 score
f1 = F1Score(task='binary', num_classes=num_classes)
f1_res = f1(preds, test_target)
print("F1 Score:", f1_res)

# Compute AUROC
auroc = AUROC(task="binary")
res = auroc(probs, test_target)
print("AUROC:", res)

# Save scores to a .txt file
scores_txt_path = savepath + "scores.txt"
with open(scores_txt_path, "w") as f:
    for metric, score in (scores[0]).items():
        f.write(f"{metric}: {score}\n")

# Save scores to a .csv file
scores_csv_path = savepath + "scores.csv"
scores_df = pd.DataFrame(list((scores[0]).items()), columns=["Metric", "Value"])
scores_df.to_csv(scores_csv_path, index=False)