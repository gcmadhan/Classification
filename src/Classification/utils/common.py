import os
import yaml
from pathlib import Path
from Classification.logging import logger
from Classification.constants import Model_path
import pandas as pd
from box import ConfigBox
from box.exceptions import BoxValueError
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectPercentile, chi2
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
import joblib

def read_yaml(path_to_yaml: Path)-> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
        
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

def create_pipeline(name: str,transformers: list):
    pipe = list()
    for i in range(len(transformers)):
        name=name+str(i)
                
        pipe.append((name,eval(transformers[i])))
    return pipe

def modeling(model, df, preprocessor, target, save=False):
    result=[]
    logger.info("Saving Model initiated")
    print(save)
    ml_pipe = Pipeline([
            ('Preprocessor',preprocessor),
            ('Model',model)
        ])
        #set_config(display='diagram')
    X= df.drop(columns=[target], axis=1)
    y= df[target]
    
    if save==False:
        X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, random_state=42,shuffle=True)
        #print(X_train.columns)
        #print(y_train)
        ml_pipe.fit(X_train, y_train)
    #print("predicted values \n")
        y_pred=ml_pipe.predict(X_test)

        acc=accuracy_score(y_pred, y_test)
        rec=recall_score(y_pred,y_test)
        pre=precision_score(y_pred,y_test)
        return(model,rec, pre, acc)
    elif save==True:
        ml_pipe.fit(X, y)
        joblib.dump(ml_pipe,os.path.join(Model_path,"Model.pkl"))
        logger.info("Model saved successfully")
        return(ml_pipe)







