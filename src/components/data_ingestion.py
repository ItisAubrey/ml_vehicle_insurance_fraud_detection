import os
import sys

from src.components.data_transformation import start_preprocessing, DataTransformationConfig
from src.components.feature_selection import FeatureSelectionConfig, feature_selection
from src.components.model_trainer import model_trainer, ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook_experiments/data/fraud_oracle.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    # Call the DataTransformation function
    X_train, y_train, X_test,y_test  = start_preprocessing(train_data, test_data, config=DataTransformationConfig())
    # Call the Feature Selection function
    final_X_train, final_X_test, selected_features = feature_selection(X_train, y_train, X_test, config=FeatureSelectionConfig())

    trained_model=model_trainer(final_X_train, final_X_test, y_train, y_test, config=ModelTrainerConfig())






