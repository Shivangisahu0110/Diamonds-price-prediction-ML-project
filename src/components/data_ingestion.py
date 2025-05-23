import os 
import sys
from src.logger import logging 
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


## Initialize the data ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts","train.csv") # to create folder and file 
    test_data_path = os.path.join("artifacts","test.csv")
    raw_data_path = os.path.join("artifacts","raw.csv")


## Create a data ingestion class

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion has started")

        try:
            df= pd.read_csv(os.path.join("notebook/data","gemstone.csv"))
            logging.info("Dataset read as pandas Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok= True)# to create directory

            df.to_csv(self.ingestion_config.raw_data_path, index= False) # to store dataframe in csv format in raw data path file

            logging.info("Train Test Split")
            train_set,test_set = train_test_split(df,test_size=30,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index =False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False, header = True)

            logging.info("Data Ingestion has completed")

            return(
                self.ingestion_config.train_data_path, #feature engineering read from here for further process
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Error occured in Data Ingestion config")

        