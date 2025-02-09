from sklearn.ensemble import RandomForestClassifier

from joblib import dump
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)
import numpy as np
import pandas as pd
import os
import time
import preprocessing.ids_preprocessing as ids_preprocessing
import data.cse_ids_dataset as cse_ids_dataset


# get dataset
dirpath = "Processed_datasets/CSE-CIC-IDS2018-AWS/cleaned/"
savepath = "Models/ids2018/randomforest-" + time.strftime('%Y%m%d-%H%M') + "/"
proc_cols = cse_ids_dataset.COLS_SLIM
proc_types = cse_ids_dataset.COLS_SLIM_TYPES

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
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

###preprocessing

# convert timestamps
times_features = ["dow", "hour", "minute"]
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Replacing 'Timestamp' column with {times_features}...")
ids_preprocessing.convert_timestamps(df, times_features)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

# binarize labels for benign/malicious traffic
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Processing Label column for binary classifier...")
ids_preprocessing.binarize_label(df)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

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
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

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
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

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
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")
# encoding test set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Encoding test set...")
# documentation suggests not passing y for test data
X_test = encoder.transform(X=X_test)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

###training model

# Create the RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=100,  # Numero di alberi
    max_depth=6,      # Profondità degli alberi
    n_jobs=-1,  # Utilizza tutti i core disponibili
)

# Fit the model
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Training...")
start_training_time = time.time()
model.fit(X_train, y_train)
end_training_time = time.time()
training_time = end_training_time - start_training_time
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

os.makedirs(savepath)
dump(model, savepath + f"randomForest.joblib")

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
results_df.to_csv(savepath + f"scores.csv", index=False)

######## test model with different weights ########

num_pos = np.sum(y_train == 1)
num_neg = np.sum(y_train == 0)

class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in (y_train.astype(int)).to_numpy()])

# Create the RandomForestClassifier for sample_weight on fit
model = RandomForestClassifier(
    n_estimators=100,  # Numero di alberi
    max_depth=6,      # Profondità degli alberi
    n_jobs=-1,  # Utilizza tutti i core disponibili
)

# Fit the model
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Training...")
start_training_time = time.time()
model.fit(X_train, y_train, sample_weight=samples_weight)
end_training_time = time.time()
training_time = end_training_time - start_training_time
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

dump(model, savepath + f"randomForest_model_sample_weights.joblib")

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
results_df.to_csv(savepath + f"randomForest_model_sample_weights.csv", index=False)

# Create the RandomForestClassifier with balanced class weights
model = RandomForestClassifier(
    n_estimators=100,  # Numero di alberi
    max_depth=6,      # Profondità degli alberi
    n_jobs=-1,  # Utilizza tutti i core disponibili
    class_weight="balanced"
)

# Fit the model
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Training...")
start_training_time = time.time()
model.fit(X_train, y_train)
end_training_time = time.time()
training_time = end_training_time - start_training_time
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

dump(model, savepath + f"randomForest_balanced.joblib")

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
results_df.to_csv(savepath + f"randomForest_model_balanced.csv", index=False)