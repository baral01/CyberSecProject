from xgboost import XGBClassifier
#from sklearn.preprocessing import TargetEncoder
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
import pandas as pd
from data.ton_iot_dataset import TonIotDataset
import time
import os

# get dataset
filepath ="Processed_datasets/TON_IoT/Network_dataset_ts_extracted.csv"
ton = TonIotDataset(filepath)
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


#separate features from label
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Separating label from features...")
y = df['label']
df.drop(columns=['label'], inplace=True)
X = df
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

#separate categorical and numerical features
NUMERICAL_FEATURES = X.select_dtypes(include='number').columns.tolist()
CATEGORICAL_FEATURES = X.select_dtypes(exclude='number').columns.tolist()

print(f"Numericals: {len(NUMERICAL_FEATURES)}; Categoricals: {len(CATEGORICAL_FEATURES)}")

#split into train and test set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Splitting train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

#target encoding of categorical features
encoder = ce.TargetEncoder(verbose=1, cols=CATEGORICAL_FEATURES, return_df=True)
#encoding train set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Encoding train set...")
encoder.fit(X=X_train, y=y_train)
X_train = encoder.transform(X=X_train, y=y_train)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")
#encoding test set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Encoding test set...")
encoder.fit(X=X_test, y=y_test)
#documentation suggests not passing y for test data
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
savepath = "Models/ton/xgboost-" + time.strftime('%Y%m%d-%H%M') + "/"
os.makedirs(savepath)
model.save_model(savepath + "xgboost_model.json")
scores = model.evals_result()
print("XGBoost evaluation results:")
print(scores)

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

