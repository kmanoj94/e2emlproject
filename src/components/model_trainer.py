import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from catboost import CatBoostRegressor
from xgboost import XGBRFRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
                              )
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbours Classifier":KNeighborsRegressor(),
                "XGBClassifier":XGBRFRegressor(),
                "CatBoost Classifier":CatBoostRegressor(),
                "AdaBoost Classifier":AdaBoostRegressor()
            }

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                               models=models)
            
            best_moedl_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_moedl_score)]

            best_model = models[best_model_name]

            if best_moedl_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_scor = r2_score(y_test,predicted)

            return r2_scor


        except Exception as e:
            raise CustomException(e,sys)
        
            

