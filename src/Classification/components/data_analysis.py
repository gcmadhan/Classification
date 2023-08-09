from Classification.entity import Outputfile
from pandas_profiling import ProfileReport
from Classification.logging import logger
import pandas as pd

class DataValidation:
    def __init__(self, config: pd.DataFrame, output_path: Outputfile):
        self.config=config
        self.Output_path = output_path.path
        print(self.Output_path)
    
    def data_anlaysis(self):

        df = self.config
        print("Data Frame Information \n")
        print(df.info)

        print("Statistica description of data \n")
        print(df.describe)

        print("Types of Features")
        print(df.dtypes.astype(str).value_counts())

        logger.info("Creating dataframe profile file")
        profile = ProfileReport(df, title="Profile_report")
        print(self.Output_path)
        profile.to_file(self.Output_path+"\Profile_report.html")
        logger.info("Profiling created successfully")


       



