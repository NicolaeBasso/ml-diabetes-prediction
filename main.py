import logging
from src.data_loader import load_data
from src.model import (
    train_model,
    cross_validate_model,
    tune_hyperparameters_random
)
from src.evaluator import evaluate_across_splits
import numpy as np

# File paths
DATA_PATH = "./data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
MODEL_PATH = "./models/diabetes_rf_model.pkl"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define random states for multiple splits
random_states = [42, 100, 200]

def main():
    # Step 1: Load and preprocess the data
    logging.info("Loading data...")
    X, y = load_data(DATA_PATH)

    # Step 2: Evaluate across splits
    all_results = evaluate_across_splits(X, y, random_states)

    # Step 3: Aggregate and display results
    for model_name in ["Random Forest", "XGBoost", "LightGBM", "Logistic Regression"]:
        model_results = [r for r in all_results if r["model"] == model_name]
        mean_f1 = np.mean([r["f1_score"] for r in model_results])
        mean_auc = np.mean([r["auc_roc"] for r in model_results])
        mean_log_loss = np.mean([r["log_loss"] for r in model_results])
        logging.info(f"\n{model_name} - Mean Metrics Across Splits:")
        logging.info(f"F1-Score: {mean_f1:.4f}, AUC-ROC: {mean_auc:.4f}, Log-Loss: {mean_log_loss:.4f}")

    # Step 4: Hyperparameter tuning for Random Forest
    logging.info("Loading Random Forest model for hyperparameter tuning...")
    rf_model = train_model(X, y)
    logging.info("Tuning hyperparameters for Random Forest...")
    best_rf_model, best_params = tune_hyperparameters_random(rf_model, X, y, n_iter=20)
    logging.info(f"Best Hyperparameters for Random Forest: {best_params}")

    # Step 5: Perform cross-validation on tuned Random Forest model
    logging.info("Performing cross-validation on tuned Random Forest model...")
    cv_scores = cross_validate_model(best_rf_model, X, y)
    logging.info(f"Cross-Validation Scores: {cv_scores}")
    logging.info(f"Average F1-Score: {cv_scores.mean():.4f}")

if __name__ == "__main__":
    main()
