import joblib
import pandas as pd
import numpy as np
import logging

# File path for the saved model
MODEL_PATH = "./models/random_forest_split_42.pkl"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Feature names
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age",
    "Education", "Income"
]

def load_model(model_path):
    """Load the trained model from a file."""
    with open(model_path, "rb") as file:
        model = joblib.load(file)
    logging.info(f"Loaded model type: {type(model)}")
    return model

def get_user_input(features):
    """Prompt the user for input and validate the values."""
    input_data = []
    for feature in features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                input_data.append(value)
                break
            except ValueError:
                print(f"Invalid input for {feature}. Please enter a numeric value.")
    return input_data

def predict_for_person_with_probabilities(model, input_data, feature_names):
    """
    Predict diabetes and show probabilities for a single person's data.
    """
    input_df = pd.DataFrame([input_data], columns=feature_names)
    probabilities = model.predict_proba(input_df)[0]
    prediction = model.predict(input_df)[0]
    result = "Diabetes" if prediction == 1 else "No Diabetes"
    return result, probabilities

if __name__ == "__main__":
    # Load model
    model = load_model(MODEL_PATH)

    # Get user input
    logging.info("Enter values for the following features:")
    user_input = get_user_input(FEATURES)

    # Predict
    try:
        prediction, probabilities = predict_for_person_with_probabilities(model, user_input, FEATURES)
        print(f"\nPrediction: {prediction}")
        print(f"Probability of No Diabetes: {probabilities[0]:.2f}")
        print(f"Probability of Diabetes: {probabilities[1]:.2f}")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
