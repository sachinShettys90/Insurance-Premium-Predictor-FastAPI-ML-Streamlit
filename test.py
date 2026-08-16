from fastapi import FastAPI, path, Query
from pydantic import BaseModel, Field, computed_field
from typing import TypedDict, Literal, Annotated, Optional, List, Dict
import json

app = FastAPI()


def load_data():
    with open('patient.json', 'r', encoding='utf-8') as f
    data = json.load(f)


@app.get('/view')
def view_patientData():
    data = load_data()
    return data
