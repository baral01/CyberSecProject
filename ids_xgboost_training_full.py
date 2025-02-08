from xgboost import XGBClassifier

# from sklearn.preprocessing import TargetEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import ids_preprocessing

# get dataset
dirpath = "Processed_datasets/CSE-CIC-IDS2018-AWS/xgboost/"
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
    "Flow Duration": "int64",
    "Tot Fwd Pkts": "int64",
    "Tot Bwd Pkts": "int64",
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
    "Fwd PSH Flags": "int64",
    "Bwd PSH Flags": "int64",
    "Fwd URG Flags": "int64",
    "Bwd URG Flags": "int64",
    "Fwd Header Len": "int64",
    "Bwd Header Len": "int64",
    "Fwd Pkts/s": "float64",
    "Bwd Pkts/s": "float64",
    "Pkt Len Min": "float64",
    "Pkt Len Max": "float64",
    "Pkt Len Mean": "float64",
    "Pkt Len Std": "float64",
    "Pkt Len Var": "float64",
    "FIN Flag Cnt": "int64",
    "SYN Flag Cnt": "int64",
    "RST Flag Cnt": "int64",
    "PSH Flag Cnt": "int64",
    "ACK Flag Cnt": "int64",
    "URG Flag Cnt": "int64",
    "CWE Flag Count": "int64",
    "ECE Flag Cnt": "int64",
    "Down/Up Ratio": "float64",
    "Pkt Size Avg": "float64",
    "Fwd Seg Size Avg": "float64",
    "Bwd Seg Size Avg": "float64",
    "Fwd Byts/b Avg": "int64",
    "Fwd Pkts/b Avg": "int64",
    "Fwd Blk Rate Avg": "int64",
    "Bwd Byts/b Avg": "int64",
    "Bwd Pkts/b Avg": "int64",
    "Bwd Blk Rate Avg": "int64",
    "Subflow Fwd Pkts": "int64",
    "Subflow Fwd Byts": "int64",
    "Subflow Bwd Pkts": "int64",
    "Subflow Bwd Byts": "int64",
    "Init Fwd Win Byts": "int64",
    "Init Bwd Win Byts": "int64",
    "Fwd Act Data Pkts": "int64",
    "Fwd Seg Size Min": "int64",
    "Active Mean": "float64",
    "Active Std": "float64",
    "Active Max": "float64",
    "Active Min": "float64",
    "Idle Mean": "float64",
    "Idle Std": "float64",
    "Idle Max": "float64",
    "Idle Min": "float64",
    "Label": "int8",
}
files = os.listdir(dirpath)
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Loading dataset...")
frames = list()
for f in files:
    print(f"Loaded file {f}")
    partial_df = pd.read_csv(dirpath + f, sep=",", usecols=proc_cols)
    frames.append(partial_df)
print("Combining frames...")
df = pd.concat(frames)
del frames
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

###preprocessing

# convert timestamps
times_features = ["dow", "hour", "minute"]
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Replacing 'Timestamp' column with {times_features}...")
ids_preprocessing.convert_timestamps(df, times_features)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

# binarize labels for benign/malicious traffic
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Processing Label column for binary classifier...")
ids_preprocessing.binarize_label(df)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

# replace inf values and deal with NaN values by replacing them with 0
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

# separate features from label
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Separating label from features...")
y = df["Label"]
df.drop(columns=["Label"], inplace=True)
X = df
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

# separate categorical and numerical features
NUMERICAL_FEATURES = X.select_dtypes(include="number").columns.tolist()
CATEGORICAL_FEATURES = X.select_dtypes(exclude="number").columns.tolist()

print(
    f"Numericals: {len(NUMERICAL_FEATURES)}; Categoricals: {len(CATEGORICAL_FEATURES)}"
)

# split into train and test set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Splitting train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

# target encoding of categorical features
encoder = ce.TargetEncoder(verbose=1, cols=CATEGORICAL_FEATURES, return_df=True)
# encoding train set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Encoding train set...")
encoder.fit(X=X_train, y=y_train)
X_train = encoder.transform(X=X_train, y=y_train)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")
# encoding test set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Encoding test set...")
encoder.fit(X=X_test, y=y_test)
# documentation suggests not passing y for test data
X_test = encoder.transform(X=X_test)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

###training model
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Training...")

# # create model instance
model = XGBClassifier(
    learning_rate=0.3,
    objective="binary:logistic",
    eval_metric=["logloss", "error", "auc"],
    verbosity=1,
)
# fit model
start_training_time = time.time()
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
end_training_time = time.time()
training_time = end_training_time - start_training_time
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {training_time}s.")
savepath = "Models/ids2018/xgboost-" + time.strftime('%Y%m%d-%H%M') + "/"
os.makedirs(savepath)
model.save_model(savepath + "xgboost_model.json")
# scores = model.evals_result()
# print("XGBoost evaluation results:")
# print(scores)

### make predictions and compute other scores
# get predictions
start_prediction_time = time.time()
y_pred = model.predict(X_test)
end_prediction_time = time.time()
num_rows = X_test.shape[0]
infer_time = (end_prediction_time - start_prediction_time) / num_rows
# get probabilities for positive class (malicious traffic)
y_pred_proba = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

# print results to stdout
print(f"Training time: {training_time}")
print(f"Average inference time: {infer_time}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print("Confusion Matrix:")
print(f"True Negatives: {conf_matrix[0][0]}")
print(f"False Positives: {conf_matrix[0][1]}")
print(f"False Negatives: {conf_matrix[1][0]}")
print(f"True Positives: {conf_matrix[1][1]}")

# save results to files
metrics_txt_path = savepath + "results.txt"
metrics_csv_path = savepath + "scores.csv"

# Writing to a .txt file
with open(metrics_txt_path, "w") as f:
    f.write(f"Training time: {training_time}\n")
    f.write(f"Average inference time: {infer_time}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"ROC AUC: {roc_auc}\n")
    f.write("Confusion Matrix:\n")
    f.write(f"True Negatives: {conf_matrix[0][0]}\n")
    f.write(f"False Positives: {conf_matrix[0][1]}\n")
    f.write(f"False Negatives: {conf_matrix[1][0]}\n")
    f.write(f"True Positives: {conf_matrix[1][1]}\n")

# Writing to a .csv file
metrics_data = {
    "Metric": ["Training_time", "Average_inference_time", "Accuracy", "F1_Score", "Precision", "Recall", "ROC_AUC", "True_Negatives", "False_Positives", "False_Negatives", "True_Positives"],
    "Value": [training_time, infer_time, accuracy, f1, precision, recall, roc_auc, conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]]
}

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(metrics_csv_path, index=False)

######## test model with different weights ########

# Calculate the scale_pos_weight
num_pos = np.sum(y_train == 1)
num_neg = np.sum(y_train == 0)
scale_pos_weight = num_neg / num_pos
# scaling factors
scaling_factors = [0.25, 0.5, 1, 2, 4]

for i in scaling_factors:
    print(f"Running model with scale_pos_weight multiplier: {i}")
    
    # Create the XGBClassifier with scale_pos_weight
    model = XGBClassifier(
        learning_rate=0.3,
        objective="binary:logistic",
        eval_metric=["logloss", "error", "auc"],
        verbosity=1,
        scale_pos_weight=i * scale_pos_weight,
    )

    # Fit the model
    start_time = time.time()
    time_str = time.strftime("%R")
    print(f"<{time_str}> Training...")
    start_training_time = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    end_training_time = time.time()
    training_time = end_training_time - start_training_time
    end_time = time.time()
    time_str = time.strftime("%R")
    print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")
    
    model.save_model(savepath + f"xgboost_model_scale_weight_f{i*100}.json")

    # Make predictions and compute other scores
    start_prediction_time = time.time()
    y_pred = model.predict(X_test)
    end_prediction_time = time.time()
    num_rows = X_test.shape[0]
    infer_time = (end_prediction_time - start_prediction_time) / num_rows
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results to stdout
    print(f"Scale_pos_weight: {i * scale_pos_weight}")
    print(f"Number of positive samples: {num_pos}")
    print(f"Number of negative samples: {num_neg}")
    print(f"Training time: {training_time}")
    print(f"Average inference time: {infer_time}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"ROC AUC: {roc_auc}")
    print("Confusion Matrix:")
    print(f"True Negatives: {conf_matrix[0][0]}")
    print(f"False Positives: {conf_matrix[0][1]}")
    print(f"False Negatives: {conf_matrix[1][0]}")
    print(f"True Positives: {conf_matrix[1][1]}")

    # Store results
    results = {
        "Scale_pos_weight": i * scale_pos_weight,
        "Training time": training_time,
        "Average inference time": infer_time,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "ROC AUC": roc_auc,
        "True Negatives": conf_matrix[0][0],
        "False Positives": conf_matrix[0][1],
        "False Negatives": conf_matrix[1][0],
        "True Positives": conf_matrix[1][1]
    }
    # Save results to a .csv file
    results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Score'])    
    results_df.to_csv(savepath + f"xgboost_resultsscale_weight_f{i*100}.csv", index=False)

# test model with different max_delta_step

max_delta_step_values = [1, 10, 100]

for max_delta_step in max_delta_step_values:
    print(f"Running model with max_delta_step: {max_delta_step}")
    
    # Create the XGBClassifier with max_delta_step
    model = XGBClassifier(
        learning_rate=0.3,
        objective="binary:logistic",
        eval_metric=["logloss", "error", "auc"],
        verbosity=1,
        max_delta_step=max_delta_step,
    )

    # Fit the model
    start_time = time.time()
    time_str = time.strftime("%R")
    print(f"<{time_str}> Training...")
    start_training_time = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    end_training_time = time.time()
    training_time = end_training_time - start_training_time
    end_time = time.time()
    time_str = time.strftime("%R")
    print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")
    
    model.save_model(savepath + f"xgboost_model_max_delta_step_{max_delta_step}.json")

    # Make predictions and compute other scores
    start_prediction_time = time.time()
    y_pred = model.predict(X_test)
    end_prediction_time = time.time()
    num_rows = X_test.shape[0]
    infer_time = (end_prediction_time - start_prediction_time) / num_rows
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results to stdout
    print(f"max_delta_step: {max_delta_step}")
    print(f"Training time: {training_time}")
    print(f"Average inference time: {infer_time}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"ROC AUC: {roc_auc}")
    print("Confusion Matrix:")
    print(f"True Negatives: {conf_matrix[0][0]}")
    print(f"False Positives: {conf_matrix[0][1]}")
    print(f"False Negatives: {conf_matrix[1][0]}")
    print(f"True Positives: {conf_matrix[1][1]}")

    # Store results
    results ={
        "max_delta_step": max_delta_step,
        "Training time": training_time,
        "Average inference time": infer_time,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "ROC AUC": roc_auc,
        "True Negatives": conf_matrix[0][0],
        "False Positives": conf_matrix[0][1],
        "False Negatives": conf_matrix[1][0],
        "True Positives": conf_matrix[1][1]
    }

    # Save results to a .csv file
    results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Score'])
    results_df.to_csv(savepath + f"xgboost_results__max_delta_step_{max_delta_step}.csv", index=False)