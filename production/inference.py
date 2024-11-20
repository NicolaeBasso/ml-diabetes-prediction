import pandas as pd
from src.utils import load_model

def predict(data, model_path):
    """
    Load the trained model and make predictions.
    Args:
        data (pd.DataFrame): Input data for prediction.
        model_path (str): Path to the saved model.
    Returns:
        predictions (list): Predicted classes (0 or 1).
    """
    # Load the trained model
    model = load_model(model_path)
    
    # Ensure input data matches the feature set used for training
    if isinstance(data, pd.DataFrame):
        predictions = model.predict(data)
    else:
        raise ValueError("Input data must be a pandas DataFrame.")

    return predictions
