from Classification.components import data_ingestion
from Classification.config.configuration import ConfigurationManager
from Classification.components.data_ingestion import DataIngestion
from Classification.components.data_analysis import DataValidation
from Classification.logging import logger
import pandas as pd

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion= DataIngestion(config=data_ingestion_config)
        return data_ingestion.read_file()
    
class DataValidationPipeline:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def main(self):
        output_file = ConfigurationManager().get_output_file()
        DataValidation(self.df, output_file).data_anlaysis()
        logger.info("Validation step completed")
