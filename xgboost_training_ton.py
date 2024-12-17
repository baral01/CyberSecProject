from xgboost import XGBClassifier
#from sklearn.preprocessing import TargetEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split
import pandas as pd
from data.ton_iot_dataset import TonIotDataset
import time

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
#print(y.head(10))
#print(X.head(10))

#separate categorical and numerical features
NUMERICAL_FEATURES = X.select_dtypes(include='number').columns.tolist()
CATEGORICAL_FEATURES = X.select_dtypes(exclude='number').columns.tolist()

print(f"Numericals: {len(NUMERICAL_FEATURES)}; Categoricals: {len(CATEGORICAL_FEATURES)}")

#split into train and test set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Splitting train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
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

# create model instance
bst = XGBClassifier(
    n_estimators=2, max_depth=2, learning_rate=0.1, objective="binary:logistic"
)
# fit model
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Training...")
print("X shape: " + str(X_train.shape))
print("Y shape: " + str(y_train.shape))
bst.fit(X_train, y_train)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")
# make predictions
preds = bst.predict(X_test)
print(preds)

