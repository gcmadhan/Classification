from Classification.entity import Outputfile, PreProcessing, DataIngestionConfig
from Classification.config.configuration import ConfigurationManager
from Classification.logging import logger
from Classification.constants import Model_path
import pandas as pd
import os, joblib
from pydantic import create_model, BaseModel

def create_dynamic_model(fields_dict):
    class DynamicBaseModel(BaseModel):
        pass
    
    for field_name, field_type in fields_dict.items():
        DynamicBaseModel.__annotations__[field_name] = field_type
    
    return DynamicBaseModel


class Data_Prediciton:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        #self.base_model=""
        self.config=ConfigurationManager()
    #
    #def create_base_model(self):
    #    target=self.config.get_data_ingestion_config().target_feature
#
    #    d_type = self.df.drop(columns=[target],axis=1).dtypes.apply(lambda x: x.name).to_dict()
    #    self.base_model = create_dynamic_model(d_type)
    #    print(type(self.base_model))
    #    return self.base_model
    #    
    
    def predict(self, data):
        #self.create_base_model()
        #for i,j in self.base_model.__annotations__.items():
        #    print(i,j)
        Model=joblib.load(os.path.join(Model_path,"Model.pkl"))
        
        return (Model.predict(data))
    
    def validate_input(self, df1):
        df = pd.read_csv(self.config.get_data_ingestion_config().data_file_path)
        target = self.config.get_data_ingestion_config().target_feature
        num_feat = df1.select_dtypes(include=['int','float','int64']).columns
        cat_feat = df1.select_dtypes(include=['object','bool']).columns
        anomoly_feat=[]
        logger.info("Numeric features:" +num_feat)
        logger.info("Cat features:" + cat_feat)
        for i in num_feat:
            if any(df1[i]< min(df[i])) | any(df1[i]>max(df[i])):
                anomoly_feat.append(i)
        
        for j in cat_feat:
            logger.info(f"feature: {j}")
            if not df[j].str.contains(df1[j][0]).any():
                anomoly_feat.append(j)
        #for j in cat_feat:
        #    
        #    if not df1[j].unique() in df[j].unique().to_list():
        #        anomoly_feat.append(j)
        return anomoly_feat


        