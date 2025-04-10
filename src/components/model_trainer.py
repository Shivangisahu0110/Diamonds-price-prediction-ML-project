import numpy as np
import pandas as pd
import sys,os
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.ModelTrainerConfig = ModelTrainerConfig
    
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train,y_train,X_test,y_test =( 
            train_array[:,:-1],
            train_array[:,-1],
            test_array[:,:-1],
            test_array[:,-1]

            )
        
            models ={
                'LinearRegression': LinearRegression(),
                'lasso':Lasso(),
                'Ridge':Ridge(),
                'ElasticNet':ElasticNet(),
                'DecisionTree':DecisionTreeRegressor()
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f"Model report: {model_report}")


            ## To get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f"best model found, Model Name:{best_model_name}, R2 score : {best_model_score}")
            print("\n====\n")
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(file_path= self.ModelTrainerConfig.trained_model_file_path,  obj = best_model )

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException (e,sys)
