from fastapi import FastAPI, HTTPException
from app.schemas import PredictionRequest, PredictionResponse
from app.model import model

app = FastAPI(
    title="ML Inference API",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        prediction = model.predict([request.features])
        return PredictionResponse(prediction=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")
