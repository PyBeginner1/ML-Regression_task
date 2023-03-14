import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj,evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Model Trainer Initiated')

            X_train, y_train, X_test, y_test = train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1]
            logging.info('Train and test split done')

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }

            model_report = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            best_score = max(sorted(model_report.values()))
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_score <= 0.6:
                raise CustomException('No best model found')
            
            logging.info(f'Best model found: [{best_model}] with R2 Score: [{best_score}]')

            save_obj(file_path=self.model_trainer_config.trained_model_path,
                     obj = best_model)

            predicted = best_model.predict(X_test)
            r2score = r2_score(y_test, predicted)
            
            logging.info(f"R2 Score of [{best_model_name}]: {r2score}")
            return r2score

        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)