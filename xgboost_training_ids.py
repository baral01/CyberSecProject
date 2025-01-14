from xgboost import XGBClassifier

# from sklearn.preprocessing import TargetEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data.cse_ids_dataset import CseIdsDataset
import time

# get dataset
filepath = "Processed_datasets/CSE-CIC-IDS2018-AWS/Thuesday-20-02-2018_processed.csv"
proc_cols = [
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "dow",
    "hour",
    "minute",
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
    "Flow ID": "string",
    "Src IP": "string",
    "Src Port": "int64",
    "Dst IP": "string",
    "Dst Port": "int64",
    "Protocol": "int64",
    "dow": "string",
    "hour": "int8",
    "minute": "int8",
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
ids = CseIdsDataset(filepath, columns=proc_cols, types=proc_types)
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
    X, y, test_size=0.2, random_state=42
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
# #params to be passed to the model
# params = {
#     "n_estimators": 2,
#     "max_depth": 6,
#     "learning_rate": 0.3
# }

# # create model instance
# bst = XGBClassifier(
#     n_estimators=2, max_depth=2, learning_rate=0.1, objective="binary:logistic"
# )
# # fit model
# start_time = time.time()
# time_str = time.strftime("%R")
# print(f"<{time_str}> Training...")
# print("X shape: " + str(X_train.shape))
# print("Y shape: " + str(y_train.shape))
# bst.fit(X_train, y_train)
# end_time = time.time()
# time_str = time.strftime("%R")
# print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")
# # make predictions
# preds = bst.predict(X_test)
# print(preds)

# XGBoost (different learning rate)

# learning_rate_range = np.arange(0.01, 1, 0.05)
# test_XG = []
# train_XG = []
# for lr in learning_rate_range:
#     xgb_classifier = XGBClassifier(learning_rate=lr)
#     xgb_classifier.fit(X_train, y_train)
#     train_XG.append(xgb_classifier.score(X_train, y_train))
#     test_XG.append(xgb_classifier.score(X_test, y_test))
# fig = plt.figure(figsize=(10, 7))
# plt.plot(learning_rate_range, train_XG, c='orange', label='Train')
# plt.plot(learning_rate_range, test_XG, c='m', label='Test')
# plt.xlabel('Learning rate')
# plt.xticks(learning_rate_range)
# plt.ylabel('Accuracy score')
# plt.ylim(0.6, 1)
# plt.legend(prop={'size': 12}, loc=3)
# plt.title('Accuracy score vs. Learning rate of XGBoost', size=14)
# plt.show()
# # Resolve overfitting
# new learning rate range
learning_rate_range = np.arange(0.01, 0.5, 0.05)
fig = plt.figure(figsize=(19, 17))
idx = 1
# grid search for min_child_weight
for weight in np.arange(0, 4.5, 0.5):
    time_str = time.strftime("%R")
    print(f"<{time_str}> Current weight: {weight}")
    train = []
    test = []
    for lr in learning_rate_range:
        time_str = time.strftime("%R")
        print(f"<{time_str}> Current lr: {lr}")
        xgb_classifier = XGBClassifier(eta=lr, reg_lambda=1, min_child_weight=weight)
        xgb_classifier.fit(X_train, y_train)
        train.append(xgb_classifier.score(X_train, y_train))
        test.append(xgb_classifier.score(X_test, y_test))
    fig.add_subplot(3, 3, idx)
    idx += 1
    plt.plot(learning_rate_range, train, c="orange", label="Training")
    plt.plot(learning_rate_range, test, c="m", label="Testing")
    plt.xlabel("Learning rate")
    plt.xticks(learning_rate_range)
    plt.ylabel("Accuracy score")
    plt.ylim(0.6, 1)
    plt.legend(prop={"size": 12}, loc=3)
    title = "Min child weight:" + str(weight)
    plt.title(title, size=16)
plt.show()
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")
