import os
import sys

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import dill
import warnings
warnings.filterwarnings('ignore')

def save_obj(file_path, obj):
    try:
        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        logging.info('Model Evaluation has begun')
        report = {}
        for name, model in models.items():
            model_params = param[name]

            grid = GridSearchCV(model, model_params, cv=3)
            grid.fit(X_train, y_train)
            model.set_params(**grid.best_params_)
            
            logging.info(f'Training completed for {name}')

            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            score = r2_score(y_test, prediction)
            report[name] = score
            
        logging.info(f"Report of Models: [{report}]")
        return report
    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)