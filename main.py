from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np 
from fastapi.responses import JSONResponse  # exception handler

# Load model and preprocessor
model = joblib.load('models/XGBoost_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Initialize app
app = FastAPI(title="Churn Prediction API")

# Define input schema using Pydantic
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    Tenure: int 
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float 

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        # Convert input to DataFrame with correct column order
        input_df = pd.DataFrame([data.dict()], columns=[
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ])
        # Preprocess
        input_processed = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0][1]

        # Return result
        return {
            "churn_prediction": int(prediction),
            "churn_probability": round(probability, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add exception handler if needed
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        detail=exc.detail
    )
