"""
This module defines a FastAPI application for predicting stroke likelihood.

It loads a pre-trained XGBoost model and a preprocessing pipeline
from pickle files.
The API endpoint `/predict` accepts patient data and returns the prediction.
"""

import pickle

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

with open("../models/xgb_model.pkl", "rb") as f:
    clf_xgb_tuned = pickle.load(f)

with open("../models/pipeline.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)


class InputData(BaseModel):
    """
    Represents the input data for the prediction model.

    Attributes:
        gender (str): Gender of the patient.
        age (int): Age of the patient.
        hypertension (int): Whether the patient
        has hypertension (1 for yes, 0 for no).
        heart_disease (int): Whether the patient
        has heart disease (1 for yes, 0 for no).
        ever_married (str): Whether the patient
        has ever been married ("Yes" or "No").
        work_type (str): Type of work the patient does.
        residence_type (str): Type of residence the
        patient lives in ("Urban" or "Rural").
        avg_glucose_level (float): Average glucose level of the patient.
        bmi (float): Body Mass Index of the patient.
        smoking_status (str): Smoking status of the patient.
    """
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


app = FastAPI()


@app.post("/predict")
async def predict(data: InputData):
    """
    Predict the likelihood of stroke based on the input data.

    Args:
        data (InputData): Input data containing features for prediction.

    Returns:
        dict: A dictionary containing the prediction.
              Example: {"prediction": 1}
    """

    input_df = pd.DataFrame([data.dict()])

    preprocessed_data = loaded_pipeline.transform(input_df)

    feature_names = ['age_binned', 'gender', 'hypertension',
                     'heart_disease', 'ever_married', 'work_type',
                     'residence_type', 'smoking_status', 'age',
                     'avg_glucose_level', 'bmi', 'age*gender',
                     'bmi*gender', 'hypertension*age_binned',
                     'heart_disease*age_binned', 'smoking*age_binned',
                     'avg_glucose*age']
    preprocessed_df = pd.DataFrame(preprocessed_data, columns=feature_names)

    prediction = clf_xgb_tuned.predict(preprocessed_df)[0]

    return {"prediction": prediction}
