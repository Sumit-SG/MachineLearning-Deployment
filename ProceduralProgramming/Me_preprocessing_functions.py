# to handle datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# load data
def load_data(DATA_PATH):
    return pd.read_csv(DATA_PATH)


# split data into train and test
def data_split(data,target):

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(target, axis=1),
        data[target],
        test_size=0.2,
        random_state=0)
    return X_train, X_test, y_train, y_test

# Extract only letter and drop number from variable cabin
def extract_letter(line):
    try:
        for l in line:
            break
        return l
    except:
        return np.nan

# add missing indicator to numeric variable
def add_missing_indicator(data,var):
    return np.where(data[var].isnull(), 1, 0)

# Impute missing values
def impute_na(data,var,impute_value="Missing"):
    return data[var].fillna(impute_value)

# Remove rare labels in categorical variables
def remove_rare_labels(data,var,frequency_ls):
    return np.where(data[var].isin(frequency_ls), data[var], 'Rare')
# Create dummy variables
def create_dummy(data,var):
    return pd.get_dummies(data=data, columns=var,drop_first=True)

# Check if all variables are present in the dataset, particularly dummy variables
def check_variables(data,column_names):
    for var in column_names:
        if var not in data.columns:
            data[var] = 0
    data = data[column_names]
    return data

def scale_features(data,scaler_file):
    scaler = joblib.load(scaler_file)
    return scaler.transform(data)

def train_model(data,target,model_path):
    log_reg = LogisticRegression(tol=0.0005, random_state=0)
    log_reg.fit(data, target)
    joblib.dump(log_reg, model_path)

def predict_model(data,model_path):
    model = joblib.load(model_path)
    pred_acc = model.predict(data)
    pred_roc = model.predict_proba(data)
    return pred_acc, pred_roc










