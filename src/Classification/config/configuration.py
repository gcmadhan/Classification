from Classification.utils.common import read_yaml
from Classification.entity import DataIngestionConfig
from Classification.constants import *

class ConfigurationManager:
    def __init__(
            self,
            param_file_path = PARAM_FILE_PATH):
        
        self.files = read_yaml(PARAM_FILE_PATH)
        

    def get_data_ingestion_config(self)->DataIngestionConfig:
        params = self.files.Input_file.file_path
        return DataIngestionConfig(params)