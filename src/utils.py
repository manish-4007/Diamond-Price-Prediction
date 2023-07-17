from src.exception import CustomException
from src.logger import logging
import os,sys
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok = True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
            logging.info('Pickle file created')
    except Exception as e:
        raise CustomException(e,sys.exc_info())
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load( file_obj)
        
    except Exception as e:
        logging.info(f'Exception occured while loadiing file object {file_path}')
        raise CustomException(e,sys.exc_info())
    
def evaluate_model(X_train,y_train,X_test,y_test,models):

    try:
        model_list=[]
        report ={}

        for i in range(len(list(models))):
            model=list(models.values())[i]
            model.fit(X_train,y_train)

            #Make Predictions
            y_pred=model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2_square = r2_score(y_test, y_pred)
            
            report[list(models.keys())[i]] = r2_square

            logging.info(list(models.keys())[i])
            print(f"\n{list(models.keys())[i]}")
            model_list.append(list(models.keys())[i])

            logging.info('Model Training Performance')
            logging.info(f"RMSE:{rmse}")
            logging.info(f"MAE:{mae}")
            logging.info(f"R2 score,{r2_square*100}")

            
            print('='*35)
            logging.info('\n\n')

        return report

    except Exception as e:
        raise CustomException(e,sys.exc_info())
    
