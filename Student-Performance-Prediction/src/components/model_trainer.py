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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.components.data_transfermation import DataTransformation
from src.components.data_igestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            logging.info(f"{X_train.shape} {y_train.shape} {X_test.shape} {y_test.shape}")
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regressor': LinearRegression(),
                'K-Neigboors Regressor': KNeighborsRegressor(),
                'CatBoosting Regressor': CatBoostRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
            }
            # simple no hyperparameters tuning
            model_report = evaluate_models(
                 X_train, y_train, X_test, y_test, models
            )

            # get best model from dict
            best_model_score = max(sorted(model_report.values()))
            models_name = list(model_report.keys())
            models_score = list(model_report.values())
            best_model_index = models_score.index(best_model_score)
            best_model_name = models_name[best_model_index]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found', sys)
            logging.info('best found model on both train and test dataset.')

            save_object(
                pkl_obj_file_path=self.model_trainer_config.trained_model_path,
                pkl_obj=best_model
            )

            prediction = best_model.predict(X_test)
            r2_score_value = r2_score(y_true=y_test, y_pred=prediction)
            return r2_score_value

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    data_ingestion = DataIngestion()
    data_transfermation = DataTransformation()
    trainer = ModelTrainer()

    train_data, test_data = data_ingestion.initiate_data_ingestion()
    train_array, test_array, preprocessor_obj = data_transfermation.initiate_data_transformation(train_data, test_data)
    score = trainer.initiate_model_trainer(train_array, test_array, preprocessor_obj)
    print(score)
