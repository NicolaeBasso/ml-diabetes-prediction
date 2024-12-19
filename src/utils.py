import joblib

def save_model(model, file_path):
    print("PREDICT: ", model.predict)

    if not hasattr(model, "predict"):
        raise ValueError(f"The object being saved is not a valid model. Type: {type(model)}")
    
    """Save the trained model to the specified file."""
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

    # Validate the saved model
    loaded_model = joblib.load(file_path)
    print(f"Loaded model type after saving: {type(loaded_model)}")

def load_model(file_path):
    """Load a saved model from the specified file."""
    return joblib.load(file_path)
