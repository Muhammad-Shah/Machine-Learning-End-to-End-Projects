import os
import sys
import json

import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.components.hyperparameters import hyper_params
from src.logger import logging


def save_object(pkl_obj_file_path, pkl_obj):
    try:
        dir_path = os.path.dirname(pkl_obj_file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(pkl_obj_file_path, 'wb') as file_obj:
            dill.dump(pkl_obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models_dict, tune=False):
    try:
        report = {}
        params = json.loads(hyper_params)
        models = list(models_dict.values())
        models_keys = list(models_dict.keys())
        i = 0
        for model in models:
            if tune:
                param_grid = params[models_keys[i]]
                gs = GridSearchCV(model, param_grid=param_grid, cv=3)
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_data_model_score = r2_score(y_train, y_train_pred)
            test_data_model_score = r2_score(y_test, y_test_pred)
            logging.info(f'{train_data_model_score} {test_data_model_score}')
            report[models_keys[i]] = test_data_model_score
            i += 1

        logging.info(f'{report}')
        
        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_pkl_oject(pkl_file_path):
    try:
        with open(pkl_file_path, 'rb') as file_obj:
            return dill.load(file=file_obj)
    except Exception as e:
        raise CustomException(e, sys)  