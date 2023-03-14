import pandas as pd
import numpy as np
import sys 
import os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from pathlib import Path

PATH = os.path.join(os.getcwd(), Path(r'artifacts\raw.csv'))

@dataclass
class DataTransformationConfig:
    preprocessed_obj_file_path = os.path.join('artifacts', 'preprocessed.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            df = pd.read_csv(PATH)

            numerical_features = [feature for feature in df.columns if df[feature].dtype != 'object']
            numerical_features.remove('math_score')
            categorical_features = [feature for feature in df.columns if df[feature].dtype == 'object']

            num_pipeline = Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Numerical Columns: {numerical_features}')
            logging.info(f'Categorical Columns: {categorical_features}')

            preprocessor = ColumnTransformer(
                [
                ('numerical', num_pipeline, numerical_features),
                ('categorical', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Data Transformation Initiated')
            train_df =  pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train and Test obtained')

            preprocessor_obj = self.get_data_transformer()
            logging.info('Obtained Preprocessor for Train and Test')

            target_col = 'math_score'

            train_X = train_df.drop(target_col, axis = 1)
            train_y = train_df[target_col]

            test_X = test_df.drop(target_col, axis = 1)
            test_y = test_df[target_col]

            logging.info('Applying preprocessor on train and test')

            input_feature_train_array = preprocessor_obj.fit_transform(train_X)
            input_feature_test_array = preprocessor_obj.fit_transform(test_X)

            train_array = np.c_[input_feature_train_array, np.array(train_y)]
            test_array = np.c_[input_feature_test_array, np.array(test_y)]

            logging.info('Saving Preprocessor')

            save_obj(
                file_path = self.data_transformation_config.preprocessed_obj_file_path,
                obj = preprocessor_obj
            )

            logging.info('Data Transformation complete')
            logging.info('\n')
            return train_array, test_array, self.data_transformation_config.preprocessed_obj_file_path

        except Exception as e:
            raise CustomException(e,sys)


