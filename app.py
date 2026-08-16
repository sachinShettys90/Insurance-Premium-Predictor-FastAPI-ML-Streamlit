'''
Run this app.py using this command  
uvicorn Project_FastAPI_With_MLModel.app.py:app --reload
'''

from fastapi import FastAPI, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Annotated, Literal, Dict, List, Optional
import pickle
import pandas as pd


# import the ml model

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)


app = FastAPI()


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

    @computed_field
    @property
    def bmi(self) -> float:
        return round(self.weight/(self.height**2), 2)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return 'high'
        elif self.smoker or self.bmi > 27:
            return "medium"
        else:
            return "low"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 60:
            return "middle_aged"
        return "senior"

    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3


@app.post('/predict')
def predict_premium(data: UserInput_PydanticModel):
    input_df = pd.DataFrame([{
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }])

    prediction = model.predict(input_df)[0]

    return JSONResponse(status_code=200, content={'prediction_category': prediction})
