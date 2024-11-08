import pandas as pd
import logging
from dataclasses import dataclass
import os
from src.utils import identify_binary_columns, label_encode_binary_columns, map_ordinal_features, \
    change_dtype_to_string, one_hot_encode, drop_useless_columns, replace_outliers_with_nan, apply_smote, \
    train_lightgbm_imputation_model, apply_imputation_model

@dataclass
class DataTrasformationConfig:
    X_train_path: str=os.path.join('artifacts',"X_train.csv")
    y_train_path: str=os.path.join('artifacts',"y_train.csv")
    X_test_path: str=os.path.join('artifacts',"X_test.csv")
    y_test_path: str=os.path.join('artifacts',"y_test.csv")
    imputation_model_path: str=os.path.join('artifacts',"imputation_model.pickle")

@dataclass
class DataTransformationConfig:
    X_train_path: str = os.path.join('artifacts', "X_train.csv")
    y_train_path: str = os.path.join('artifacts', "y_train.csv")
    X_test_path: str = os.path.join('artifacts', "X_test.csv")
    y_test_path: str = os.path.join('artifacts', "y_test.csv")

def start_preprocessing(train_path, test_path, config=DataTransformationConfig()):

    # Read data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    logging.info("Read train and test data completed")

    target_column_name = 'FraudFound_P'
    X_train = train_df.drop(columns=[target_column_name], axis=1)
    y_train = train_df[target_column_name]

    X_test = test_df.drop(columns=[target_column_name], axis=1)
    y_test = test_df[target_column_name]

    # Binary Column Identification and Encoding
    binary_columns = identify_binary_columns(X_train)
    X_train = label_encode_binary_columns(X_train, binary_columns)
    X_test = label_encode_binary_columns(X_test, binary_columns)
    logging.info("Successfully encoded binary columns.")

    # Ordinal Feature Mapping
    ordinal_mappings = {
        'VehiclePrice': {'more than 69000': 1, '20000 to 29000': 0, '30000 to 39000': 0, 'less than 20000': 1, '40000 to 59000': 1, '60000 to 69000': 0},
        'AgeOfVehicle': {'new': 2, '2 years': 0, '3 years': 2, '4 years': 2, '5 years': 1, '6 years': 1, '7 years': 0, 'more than 7': 0},
        'BasePolicy': {'Liability': 0, 'Collision': 1, 'All Perils': 2}
    }
    X_train = map_ordinal_features(X_train, ordinal_mappings)
    X_test = map_ordinal_features(X_test, ordinal_mappings)
    logging.info("Mapped ordinal features")

    # Data Type Conversion
    dtype_change_string = ['RepNumber', 'Deductible', 'Year']
    X_train = change_dtype_to_string(X_train, dtype_change_string)
    X_test = change_dtype_to_string(X_test, dtype_change_string)

    # One-Hot Encoding
    onehot_encoding_columns = ['Make', 'MonthClaimed', 'MaritalStatus', 'PolicyType', 'VehicleCategory', 'RepNumber', 'Deductible', 'Days_Policy_Accident', 'Days_Policy_Claim', 'PastNumberOfClaims', 'AgeOfPolicyHolder', 'NumberOfSuppliments', 'AddressChange_Claim', 'NumberOfCars', 'Year']
    X_train = one_hot_encode(X_train, onehot_encoding_columns)
    X_test = one_hot_encode(X_test, onehot_encoding_columns)
    logging.info("Applied one-hot encoding")

    # Drop Useless Columns
    useless_columns = ['Month', 'WeekOfMonth', 'DayOfWeek', 'DayOfWeekClaimed', 'WeekOfMonthClaimed', 'PolicyNumber']
    X_train = drop_useless_columns(X_train, useless_columns)
    X_test = drop_useless_columns(X_test, useless_columns)
    logging.info("Dropped useless columns")

    # Outlier Handling and Imputation
    X_train = replace_outliers_with_nan(X_train, 'Age')
    X_test = replace_outliers_with_nan(X_test, 'Age')

    # Train LightGBM model on non-missing Age values from X_train
    age_imputation_model = train_lightgbm_imputation_model(X_train, 'Age')
    X_train = apply_imputation_model(X_train, age_imputation_model, 'Age')
    X_test = apply_imputation_model(X_test, age_imputation_model, 'Age')
    X_train['Age'] = X_train['Age'].apply(lambda x: round(x))
    X_test['Age'] = X_test['Age'].apply(lambda x: round(x))

    # SMOTE Oversampling
    X_train, y_train = apply_smote(X_train, y_train)

    # Save the processed data to CSV
    X_train.to_csv(config.X_train_path, index=False, header=True)
    y_train.to_csv(config.y_train_path, index=False, header=True)
    X_test.to_csv(config.X_test_path, index=False, header=True)
    y_test.to_csv(config.y_test_path, index=False, header=True)
    logging.info("Transformation of the data is completed")

    return (
        config.X_train_path,
        config.y_train_path,
        config.X_test_path,
        config.y_test_path,
    )













