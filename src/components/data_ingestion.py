import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path : str = os.path.join('artifacts','raw.csv')
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def inititate_data_inestion(self):
        logging.info('Data Ingestion Initiated')
        try:
            df = pd.read_csv('notebook\data\StudentsPerformance.csv')

            new_cols = [col.replace(' ','_').replace('/','_') for col in df.columns]
            df.columns = new_cols

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f'Raw data stored at: [{self.ingestion_config.raw_data_path}]')

            logging.info('Train-Test split Initiated')
            train, test = train_test_split(df, test_size=0.2,random_state=1)

            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion completed")
            logging.info("\n")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e,sys) from e
        

if __name__ == '__main__':
    data = DataIngestion()
    train, test = data.inititate_data_inestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train, test)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)