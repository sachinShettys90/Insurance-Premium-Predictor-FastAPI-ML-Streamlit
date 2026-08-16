'''
Run this app.py using this command  
uvicorn app_withfeature_engineering:app --reload

NOTE: Here we are using the feature_engineering.py instead of adding the computedfields in the pydantic model
'''

from fastapi import FastAPI, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal, Dict, List, Optional
import pickle
import pandas as pd
from MLModel.feature_engineering import compute_features

# import the ml model

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'MLModel', 'model.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


app = FastAPI()


@app.get('/about')
def about():
    return {"message": "A fully functional API to Insurance Premium predictor"}


# tier cities defination
tier_1_cities = ["Mumbai", "Delhi", "Bangalore",
                 "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]


# build pydantic model to validate the incoming data
class UserInput_PydanticModel(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=120,
                              description="Age of the user")]
    weight: Annotated[float,
                      Field(..., gt=0, description="weight of the user")]
    height: Annotated[float,
                      Field(..., gt=0, lt=2.5, description="height of the user")]
    income_lpa: Annotated[float,
                          Field(..., gt=0, description="income_lpa of the user")]
    smoker: Annotated[bool, Field(..., description="User is Smoker or not")]
    city: Annotated[str, Field(..., description="user's city")]
    occupation: Annotated[Literal['retired', 'unemployed', 'business_owner', 'government_job',
                                  'private_job', 'freelancer', 'student'],
                          Field(..., description="Occupation of the user")]


@app.post('/predict')
def predict_premium(data: UserInput_PydanticModel):
    features = compute_features(
        age=data.age, weight=data.weight, height=data.height,
        income_lpa=data.income_lpa, smoker=data.smoker,
        city=data.city, occupation=data.occupation,
    )
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]

    return JSONResponse(status_code=200, content={'prediction_category': prediction})
