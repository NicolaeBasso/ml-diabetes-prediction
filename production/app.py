from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from production.inference import predict

# Initialize the FastAPI app
app = FastAPI()

# Path to the trained model
MODEL_PATH = "../models/diabetes_rf_model.pkl"

# Define the schema for input data using Pydantic
class PredictionRequest(BaseModel):
    features: list  # Example: [10.0, 1, 28.0, ...]

@app.post("/predict")
def make_prediction(request: PredictionRequest):
    """
    Endpoint to predict diabetes status from input features.
    Args:
        request (PredictionRequest): A JSON payload containing input features.
    Returns:
        dict: The predicted class (0 or 1).
    """
    try:
        # Convert the input features to a DataFrame
        data = pd.DataFrame([request.features], columns=[
            "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
            "Stroke", "HeartDiseaseorAttack", "PhysActivity",
            "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
            "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
            "Sex", "Age", "Education", "Income"
        ])
        
        # Run the prediction
        predictions = predict(data, MODEL_PATH)
        return {"prediction": int(predictions[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
