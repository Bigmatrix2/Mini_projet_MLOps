"""
API de prédiction - Modèle Iris
Auteur : Coulibaly Mohamed Abdulaziz
"""

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

app = FastAPI(
    title="Iris Prediction API",
    description="API MLOps - Coulibaly Mohamed Abdulaziz",
    version="1.0.0"
)

MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
model_data = None

@app.on_event("startup")
def load_model():
    global model_data
    try:
        model_data = joblib.load(MODEL_PATH)
        print(f"Modèle chargé depuis {MODEL_PATH}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")


class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: List[float]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_data is not None
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model_data is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    features = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]])

    pipeline = model_data["pipeline"]
    target_names = model_data["target_names"]

    prediction = int(pipeline.predict(features)[0])
    probabilities = pipeline.predict_proba(features)[0].tolist()
    class_name = target_names[prediction]

    return PredictResponse(
        prediction=prediction,
        class_name=class_name,
        probabilities=probabilities
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)