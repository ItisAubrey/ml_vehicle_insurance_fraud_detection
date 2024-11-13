import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle
#%% Data Preparation Functions

def identify_binary_columns(X):
    """Identify columns with exactly two unique values."""
    return [col for col in X.columns if X[col].nunique() == 2]

def label_encode_binary_columns(df, columns):
    """Label encode binary columns in the dataframe."""
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
        print(f"Label Mapping for {col}: ", dict(zip(le.classes_, le.transform(le.classes_))))
    return df

def map_ordinal_features(df, mapping_dicts):
    """Map ordinal categorical features based on provided mappings."""
    for col, mapping in mapping_dicts.items():
        df[col] = df[col].map(mapping)
    return df

def change_dtype_to_string(df, columns):
    """Convert specified columns to string type."""
    for col in columns:
        df[col] = df[col].astype(str)
    return df

def one_hot_encode(df, columns):
    """Apply one-hot encoding to specified columns."""
    return pd.get_dummies(df, columns=columns)

def drop_useless_columns(df, columns):
    """Drop specified columns from the dataframe."""
    return df.drop(columns=columns, axis=1)

#%% LightGBM Imputation for Outlier Handling

def train_lightgbm_imputation_model(X_train, feature):
    """Train LightGBM model for imputation on a specific feature and return the trained model."""
    train_data = X_train[X_train[feature].notnull()]

    X_train_data = train_data.drop(columns=[feature], axis=1)
    y_train_data = train_data[feature]

    train_dataset = lgb.Dataset(X_train_data, label=y_train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1
    }
    model = lgb.train(params, train_dataset, num_boost_round=100)

    return model


def apply_imputation_model(df, model, feature):
    """Apply a pre-trained LightGBM model to impute missing values for a feature in a dataframe."""
    test_data = df[df[feature].isnull()]
    X_test_data = test_data.drop(columns=[feature], axis=1)

    # Predict missing values
    predictions = model.predict(X_test_data)

    # Fill in the predictions for missing values
    df.loc[df[feature].isnull(), feature] = predictions

    return df

#%% Outlier Handling

def replace_outliers_with_nan(df, column, threshold=74):
    """Replace outliers and zero values with NaN."""
    df[column] = df[column].apply(lambda x: np.nan if x == 0 or x > threshold else x)
    return df

#%% SMOTE Oversampling

def apply_smote(X, y):
    """Apply SMOTE to balance classes in the training dataset."""
    smote = SMOTE(random_state=0)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print("Before SMOTE:", X.shape, y.shape)
    print("After SMOTE:", X_resampled.shape, y_resampled.shape)
    print("After SMOTE Label Distribution:", pd.Series(y_resampled).value_counts())
    return X_resampled, y_resampled


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)