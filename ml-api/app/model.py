import joblib

MODEL_PATH = "model/model.joblib"

# Load model once at startup
model = joblib.load(MODEL_PATH)
