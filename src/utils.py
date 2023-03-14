import os
import sys

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score

import dill

def save_obj(file_path, obj):
    try:
        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        logging.info('Model Evaluation has begun')
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            score = r2_score(y_test, prediction)
            report[name] = score
            
        logging.info(f"Report of Models: [{report}]")
        return report
    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys)