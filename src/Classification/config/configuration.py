from Classification.utils.common import read_yaml
from Classification.entity import DataIngestionConfig, Outputfile, PreProcessing, Models
from Classification.constants import *
from Classification.logging import logger

class ConfigurationManager:
    def __init__(
            self,
            param_file_path = PARAM_FILE_PATH):
        logger.info(f"Reading Params file {param_file_path}")
        self.files = read_yaml(PARAM_FILE_PATH)
        
        

    def get_data_ingestion_config(self)->DataIngestionConfig:
        inp_file= DataIngestionConfig(
        data_file_path = self.files.Input_file.file_path,
        target_feature = self.files.Input_file.target_feature)
        return inp_file
    
    def get_output_file(self)->Outputfile:
        path = self.files.Output_file.path
        return Outputfile(path)
    
    def get_preprocess(self)->PreProcessing:
        pre_process=PreProcessing(
        cat_trans= self.files.Preprocessing.Categorical_transformer,
        num_trans = self.files.Preprocessing.Numerical_trasformer)
        return pre_process
    
    def get_models(self)->Models:
        models = self.files.Models
        return Models(models)