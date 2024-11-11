from sklearn.ensemble import RandomForestClassifier
from BorutaShap import BorutaShap
import xgboost as xgb
import os
from dataclasses import dataclass
import pandas as pd
import logging
import pickle


@dataclass
class FeatureSelectionConfig:
    final_X_train_path: str = os.path.join('artifacts', "X_train_final.csv")
    final_X_test_path: str = os.path.join('artifacts', "X_test_final.csv")
    selected_features_path: str = os.path.join('artifacts', "selected_features.pickle")


def feature_selection(X_train_path, y_train_path, X_test_path, config=FeatureSelectionConfig()):
    # Read data
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()  # Convert to 1D array if needed
    X_test = pd.read_csv(X_test_path)
    logging.info("Read train and test data completed")

    # Use a Random Forest model as the base
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Create an XGBoost model with GPU support
    xgboost_model = xgb.XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', use_label_encoder=False,
                                      eval_metric='logloss')

    # Apply BorutaShap using XGBoost
    feat_selector = BorutaShap(model=xgboost_model, importance_measure='shap', classification=True)

    # Fit BorutaShap (with sampling to speed it up)
    feat_selector.fit(X=X_train, y=y_train, n_trials=50, sample=0.1, train_or_test='train', normalize=True)

    # Get the selected features
    selected_features = feat_selector.accepted

    # Filter the training set to keep only the selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Save the processed data to CSV
    X_train_selected.to_csv(config.final_X_train_path, index=False, header=True)
    X_test_selected.to_csv(config.final_X_test_path, index=False, header=True)

    # Save the selected features list as a pickle file
    with open(config.selected_features_path, 'wb') as f:
        pickle.dump(selected_features, f)

    logging.info("Feature Selection is completed")

    return (
        config.final_X_train_path,
        config.final_X_test_path,
        config.selected_features_path
    )
