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
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor


from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model
# from src.components.model_trainer import ModelTrainingConfig
# from src.components.model_trainer import Modeltrainer

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class Modeltrainer :
    def __init__(self):
        self.model_trainer_Config = ModelTrainingConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data")
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "K-Neighbors Classifier": KNeighborsRegressor(),
            "XGBClassifier": XGBRegressor(),
            "CatBoosting Classifier": CatBoostRegressor(verbose=False),
            "AdaBoost Classifier": AdaBoostRegressor()
            }
            model_report: dict = evaluate_model(X_train= X_train ,y_train = y_train, X_test = X_test , y_test = y_test, models=models)

            ## to get best model score from dict
            best_model_name = max(model_report, key=model_report.get)
             ## to get best model name from dict
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException ("No best model is found")

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object (
                file_path = self.model_trainer_Config.trained_model_file_path,
                obj = best_model
            )
            predicted =best_model.predict(X_test)
            test_r2_score = r2_score(y_test,predicted)
            return test_r2_score
        except Exception as e:
            raise CustomException (e,sys)