import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model
from dataclasses import dataclass
import os,sys

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self) :
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Dependent and Taget variable from train and test array that got After Data Transformation")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
            }

            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            logging.info(f'Model Report: \n{model_report}')

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            print(f'Best Model: {best_model_name}, with R2 Score : {best_model_score}') 
            logging.info(f'Best Model: {best_model_name}, with R2 Score : {best_model_score}') 
            
            save_object(self.model_trainer_config.trained_model_file_path, best_model)


        except Exception as e:
            raise CustomException(e,sys.exc_info())