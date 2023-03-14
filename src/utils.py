import os
import sys

from src.exception import CustomException

import dill

def save_obj(file_path, obj):
    try:
        with open(file_path,'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)