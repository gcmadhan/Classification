from fastapi import FastAPI, UploadFile, File
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import joblib, os
import pandas as pd
import uvicorn
from Classification.pipeline import *
from Classification.components.data_prediction import Data_Prediciton
from Classification.config.configuration import ConfigurationManager
app = FastAPI()
df = DataIngestionPipeline().main()
  
#items = Data_Prediciton(df)
class Items(BaseModel):
    Age: int
    Gender: object
    Tenure: int
    UsageFrequency: int
    SupportCalls: int
    PaymentDelay: int
    SubscriptionType: object
    ContractLength: object
    TotalSpend: int
    LastInteraction: int
    

@app.post("/predict/")
async def get_items(items: Items):
    data = jsonable_encoder(items)
    
    
    data=pd.DataFrame.from_dict(data, orient='index').T
    
    print(data)
    
    pred = ModelPredictPipeline(data).main()
    return {"message": str(pred)}

@app.get("/train/")
async def get_items():
    df = DataIngestionPipeline().main()
    df=df.drop(columns=['CustomerID'], axis=1)
    DataValidationPipeline(df).main()
    msg = DataPreprocessingModelingPipeline(df).main()
    #ModelSavePipeline(df,1)
    return {"message": str(msg)}

@app.post("/save/")
async def get_items(modelid: int):
    df = DataIngestionPipeline().main()
    df=df.drop(columns=['CustomerID'], axis=1)
    #DataValidationPipeline(df).main()
    #msg = DataPreprocessingModelingPipeline(df).main()
    ModelSavePipeline(df,modelid).main()
    return {"message": str("Model saved successfully")}

if __name__=="__main__":
    uvicorn.run(app, port=5000, log_level="info")