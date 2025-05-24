from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("wine_model.pkl")

@app.post("/predict")
def predict(data: list):
    prediction = model.predict([data])
    return {"prediction": int(prediction[0])}
