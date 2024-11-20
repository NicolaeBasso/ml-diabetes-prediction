import joblib

def save_model(model, file_path):
    """Save the trained model to the specified file."""
    joblib.dump(model, file_path)

def load_model(file_path):
    """Load a saved model from the specified file."""
    return joblib.load(file_path)
