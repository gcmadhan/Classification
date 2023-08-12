from Classification.entity import Outputfile, PreProcessing, DataIngestionConfig
from Classification.config.configuration import ConfigurationManager
from Classification.logging import logger
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from Classification.utils.common import create_pipeline, modeling
from sklearn import set_config
import os

class Data_Preprocessing:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        config=ConfigurationManager()
        self.models = config.get_models().models
        self.target = config.get_data_ingestion_config().target_feature
        self.num_trans = config.get_preprocess().num_trans
        self.cat_trans = config.get_preprocess().cat_trans
        self.output_path = config.get_output_file().path
        self.preprocessor=""
    
 

    def modeling(self):
        df = self.df
        
        
        target= self.target
        

        num_trans = self.num_trans
        cat_trans = self.cat_trans

        model = self.models

        print(f"Numeric transformer {num_trans}\n")
        print(f"Numeric transformer {cat_trans}\n")
                
        
        
        #model=models
        preprocessor = self.preprocessing(self.df)
        result = pd.DataFrame(columns=['Model','recall_score','precision_score','accuracy_score'])
        for i,m in enumerate(model):
            result.loc[i]=modeling(eval(m), df, preprocessor, target)
            result.to_csv(os.path.join(self.output_path,"result.csv"))
        
        return result
    
    def save_model(self, model_id: int, save: bool):
        model=self.models[model_id]
        print(f"***************\n\n\n save {save} \n\n\n**************")
        model = modeling(eval(model),self.df,self.preprocessing(self.df),self.target, save)
        

    def preprocessing(self, df ):
        num_trans = self.num_trans
        cat_trans = self.cat_trans
        target= self.target
        num_feat = df.drop(columns=[target],axis=1).select_dtypes(include=['int','float','int64']).columns
        
        cat_feat = df.select_dtypes(include=['object','bool']).columns
        #cat_feat=cat_feat.remove('price')
        print(f"Numeric features are: {num_feat}")
        print(f"Categorical features are: {cat_feat}")
        
        ##Numeric transformer pipeline
        num_pipe = Pipeline(create_pipeline('Numeric',num_trans))
        
        cat_pipe = Pipeline(create_pipeline('Categorial',cat_trans))
        


        preprocessor = ColumnTransformer(transformers=[('num',num_pipe, num_feat),
                                                       ('cat',cat_pipe,cat_feat)
                                                      ], remainder='passthrough')
        return preprocessor
    

        
        



