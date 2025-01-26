import uvicorn
from fastapi import FastAPI
import pandas as pd 
from pydantic import BaseModel
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware

with open("model.pkl", "rb") as file_obj:
    model = pickle.load(file_obj)

with open("scaler.pkl", "rb") as file_obj:
    scaler = pickle.load(file_obj)

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

if __name__ == '__main__':
    uvicorn.run("app:app", host='127.0.0.1', port=8000,reload=True)