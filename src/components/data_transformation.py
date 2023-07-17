import pandas as pd
import numpy as np

from dataclasses import dataclass
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.impute import SimpleImputer ## HAndling Missing Values
from sklearn.preprocessing import StandardScaler # HAndling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Create Data Transformation Configuration for input or output information
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

# Create Data Transformation
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation started.....')
            # Segregating numerical and categorical variables
            categorical_cols =['cut','color','clarity']
            numerical_cols =['carat','depth','table','x','y','z']
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Data Transformation Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            logging.info('Data Transformation completed')

            return preprocessor


        except Exception as e:
            logging.info("Exception occured at Data Transformation Stage")
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            
            logging.info("Read Train and Test Data Completed")
            logging.info(f'Train Dataframe Head :\n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head :\n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object.......')

            preprocessing_obj = self.get_data_transformation_object()

            target_column = 'price'
            drop_column = [target_column,'id']

            # Dividing the dataset into dependent and independent dataFrames

            ##### For Training Data 
            input_feature_train_df = train_df.drop(columns = drop_column,axis =1)
            target_feature_train_df = train_df[target_column]

            ##### For Testining Data 
            input_feature_test_df = test_df.drop(columns = drop_column,axis =1)
            target_feature_test_df = test_df[target_column]

            # Data Transformation 
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info("Transforming Train and Test Data Completed")
            logging.info(f'Train Dataframe Head :\n{train_arr[:,:5]}')
            logging.info(f'Test Dataframe Head :\n{test_arr[:,:5]}')

            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessing_obj)

            logging.info('Appyling  preprocessing Train and Test data Completed ')

            return(
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
                )

        except Exception as e:
            logging.info("Exception occured at Data Transformation Stage")
            raise CustomException(e,sys.exc_info())