
from fastapi import FastAPI
from pydantic import BaseModel
import gzip
import pickle
import numpy as np

app = FastAPI()

with gzip.open("electricity_model_compressed.pkl.gz", "rb") as f:
    model = pickle.load(f)

class ElectricityInput(BaseModel):
    site_area: float
    structure_type: str
    water_consumption: float
    recycling_rate: float
    utilisation_rate: float
    air_quality_index: float
    issue_resolution_time: float
    resident_count: int

def encode_structure_type(structure):
    mapping = {
        "Residential": 0,
        "Commercial": 1,
        "Mixed-use": 2,
        "Industrial": 3
    }
    return mapping.get(structure, -1)

@app.post("/predict")
def predict(data: ElectricityInput):
    encoded_structure = encode_structure_type(data.structure_type)
    if encoded_structure == -1:
        return {"error": "Invalid structure type"}

    features = [
        data.site_area,
        encoded_structure,
        data.water_consumption,
        data.recycling_rate,
        data.utilisation_rate,
        data.air_quality_index,
        data.issue_resolution_time,
        data.resident_count
    ]

    prediction = model.predict([features])[0]
    return {"predicted_electricity_cost_usd_per_month": round(prediction, 2)}
