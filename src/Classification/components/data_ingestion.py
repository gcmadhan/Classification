from Classification.config.configuration import ConfigurationManager
from Classification.logging import logger
from Classification.entity import DataIngestionConfig
import pandas as pd
import os
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def read_file(self):
        file_path = self.config.data_file_path
        try:
            ext = os.path.splitext(file_path)[1]
            if ext ==".csv":
                df = pd.read_csv(file_path)
            #elif ext==".xlsx":
            logger.info("Reading Dataset: Successful")
            return df
        except Exception as e:
            logger.warning("Input file is not in acceptable formate {file_path}")
            logger.exception(e)
            raise e
    
            


        
    
