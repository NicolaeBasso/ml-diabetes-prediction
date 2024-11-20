import pandas as pd
from production.inference import predict

# File paths
MODEL_PATH = "../models/diabetes_rf_model.pkl"
NEW_DATA_PATH = "../data/new_test_data.csv"

def main():
    # Load the new test data
    print("Loading new test data...")
    new_data = pd.read_csv(NEW_DATA_PATH)

    # Drop target column if it exists
    if 'Diabetes_binary' in new_data.columns:
        new_data = new_data.drop('Diabetes_binary', axis=1)

    # Make predictions
    print("Running predictions...")
    predictions = predict(new_data, MODEL_PATH)

    # Save predictions
    new_data['Predicted'] = predictions
    new_data.to_csv("../results/predictions.csv", index=False)
    print("Predictions saved to '../results/predictions.csv'")

if __name__ == "__main__":
    main()
