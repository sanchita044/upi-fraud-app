import uvicorn
from fastapi import FastAPI
import pandas as pd 
from pydantic import BaseModel
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware
from datetime import date
import json

with open("model.pkl", "rb") as file_obj:
    model = pickle.load(file_obj)

with open("scaler.pkl", "rb") as file_obj:
    scaler = pickle.load(file_obj)

with open('data.json', 'r') as file:
    data = json.load(file)

fraud_db = pd.read_csv("fraud_upi_list.csv")

class UpiData(BaseModel):
    transDay: int	
    transMonth: int	
    transYear: int	
    upiNumber: int	
    transAmount: float

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.post('/predict-upi-fraud')
def UpiFraudPredictionEndpoint(data:UpiData):
      data_dict = data.dict()
      transDay= data_dict['transDay']
      transMonth= data_dict['transMonth']
      transYear= data_dict['transYear']
      upiNumber= data_dict['upiNumber']
      transAmount= data_dict['transAmount']
      test = pd.DataFrame([{
            'trans_day': transDay,'trans_month': transMonth,
            'trans_year': transYear,'upi_number': upiNumber,'trans_amount': transAmount
      }])
      scaleTest=scaler.transform(test)
      predict=model.predict(scaleTest)
      if(predict==[1]):
         status="FRAUD"
      else:
         status="NOT FRAUD"
      return {
                'prediction': status
            }

@app.get('/get-upi-data')
def getUpiDataEndpoint():
    todays_date = date.today()
    predict=[]
    for item in data:
        test = pd.DataFrame([{
            'trans_day': todays_date.day,'trans_month': todays_date.month,
            'trans_year': todays_date.year,'upi_number': item["Phone Number"],'trans_amount': item["Amount"]
        }])
        scaleTest=scaler.transform(test)
        result=model.predict(scaleTest)
        if(result==[1]):
            status="FRAUD"
        else:
            status="NOT FRAUD"
        predictionData = {"upi_number":item["Phone Number"],"trans_amount": item["Amount"],"status":status}
        predict.append(predictionData)
      
    return {
        'data': predict
    }


class UPIRequest(BaseModel):
    upi_id: str

@app.post("/check_upi")
async def check_upi(data: UPIRequest):
    upi_id = data.upi_id.strip()
    is_fraud = upi_id in fraud_db["upi_id"].values
    return {"upi_id": upi_id, "fraud": is_fraud}


if __name__ == '__main__':
    uvicorn.run("app:app", host='127.0.0.1', port=8000,reload=True)