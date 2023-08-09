from Classification.entity import Outputfile, PreProcessing, DataIngestionConfig
from Classification.config.configuration import ConfigurationManager
from Classification.logging import logger
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import set_config
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class Data_Preprocessing:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def preprocessing_modeling(self):
        df = self.df
        config=ConfigurationManager()
        
        target= config.get_data_ingestion_config().target_feature
        

        num_trans = eval(config.get_preprocess().num_trans[0])
        cat_trans = eval(config.get_preprocess().cat_trans[0])

        model = eval(config.get_models().models[0])

        print(f"Numeric transformer {num_trans}\n")
        print(f"Numeric transformer {cat_trans}\n")
        num_feat = df.drop(columns=[target],axis=1).select_dtypes(include=['int','float','int64']).columns
        
        cat_feat = df.select_dtypes(include=['object','bool']).columns
        #cat_feat=cat_feat.remove('price')
        print(f"Numeric features are: {num_feat}")
        print(f"Categorical features are: {cat_feat}")
        
        ##Numeric transformer pipeline
        num_pipe = Pipeline([("scalar", num_trans)])
        cat_pipe = Pipeline([("OneHotencoding",cat_trans)])


        preprocessor = ColumnTransformer(transformers=[('num',num_pipe, num_feat),
                                                       ('cat',cat_pipe,cat_feat)
                                                      ], remainder='passthrough')
        
        
        
        #model=models

        ml_pipe = Pipeline([
            ('Preprocessor',preprocessor),
            ('Model',model)
        ])
        #set_config(display='diagram')
        X= df.drop(columns=[target], axis=1)
        y= df[target]
        print(preprocessor.fit(X))
        X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, random_state=42,shuffle=True)
        #print(X_train.columns)
        #print(y_train)
        ml_pipe.fit(X_train, y_train)
        print("predicted values \n")
        y_pred=ml_pipe.predict(X_test)

        print("Metrics \n")
        print(confusion_matrix(y_pred, y_test))
        print("\n\n\n")

        print(classification_report(y_pred, y_test))
        print("\n\n\n")
        print(accuracy_score(y_pred, y_test))
        
        
        
        



