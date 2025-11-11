import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
import json
import pandas as pd

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        data = json.loads(raw.decode("utf-8"))
        df = pd.DataFrame([data])
        model = joblib.load("diabetes_model.joblib")
        prediction = model.predict(df)
        return {"prediction": prediction.tolist()}
    except HTTPException:
        raise {"message": "model prediction failed"}