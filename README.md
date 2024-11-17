# ml_vehicle_insurance_fraud_detection

This project tackles vehicle insurance claim fraud detection using machine learning. It provides an end-to-end solution for identifying potentially fraudulent claims.

## Data Source:
The project leverages a vehicle claim fraud detection dataset publicly available on Kaggle: https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection

## Project Structure:
### Notebook Experiments:

- Exploratory Data Analysis (EDA): In-depth analysis of the dataset to understand its characteristics, identify patterns, and uncover potential insights.
- Model Training: Experimentation with various machine learning algorithms to determine the most effective model for fraud detection.
This project contains the notebook experiment for the Exploratory Analysis, and model training with different types of machine learning models


### Source Code:

The source code is organized into four modular components:

- Data Ingestion: Handles the process of loading the dataset into the system.
- Data Transformation: Prepares the data for modeling by cleaning, imputing missing values, and feature engineering.
- Feature Selection: Identifies the most relevant features for accurate prediction using BorutaShap method.
- Model Training: Trains and evaluates machine learning models on the prepared dataset.

Flask Application
- A Flask-based web application is developed to deploy the best-performing model. This application allows users to input claim information, and the model will predict whether the claim is likely to be fraudulent.