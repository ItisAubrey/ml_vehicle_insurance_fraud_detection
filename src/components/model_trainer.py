# Based on the exploration in the notebook, gonna apply the best model random forest for this dataset
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score
import os
from dataclasses import dataclass
import pandas as pd
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score
import os
from dataclasses import dataclass
import pandas as pd

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

def model_trainer(final_X_train_path, final_X_test_path, y_train_path, y_test_path, config=ModelTrainerConfig()):
    # Load data
    final_X_train = pd.read_csv(final_X_train_path)
    final_X_test = pd.read_csv(final_X_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()  # Convert to 1D array if needed
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Initialize the RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    # Define the parameter grid for Grid Search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Setup Grid Search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)

    # Fit the model
    grid_search.fit(final_X_train, y_train)

    # Get the best estimator
    best_rf = grid_search.best_estimator_

    # Predict probabilities for ROC-AUC calculation
    y_pred_prob = best_rf.predict_proba(final_X_test)[:, 1]  # Use probabilities for positive class
    auc_score = roc_auc_score(y_test, y_pred_prob)
    auc_score = round(auc_score, 3)

    print("Best Parameters:", grid_search.best_params_)
    print("ROC-AUC Score:", auc_score)

    # Save the best model
    with open(config.trained_model_file_path, 'wb') as f:
        pickle.dump(best_rf, f)

    return config.trained_model_file_path


