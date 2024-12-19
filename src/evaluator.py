import logging
import os  # Import os to access environment variables
from dotenv import load_dotenv  # Import dotenv
from sklearn.metrics import roc_auc_score, log_loss
from src.data_loader import split_data
from src.model import (
    train_model,
    train_xgboost,
    train_lightgbm,
    train_logistic_regression,
    evaluate_model
)
from src.utils import save_model
from src.visualizations import plot_confusion_matrix, plot_feature_importance

load_dotenv()

def evaluate_across_splits(X, y, random_states):
    """
    Evaluate models across multiple train-test splits.

    Args:
        X: Features dataset.
        y: Target labels.
        random_states: List of random states for splitting.

    Returns:
        list: A list of results containing metrics for each split and model.
    """
    results = []

    # Read SHOW_PLOTS from the environment or default to 1
    show_plots = os.getenv("SHOW_PLOTS", "1") == "1"

    for random_state in random_states:
        logging.info(f"\nEvaluating models with random_state={random_state}...")

        # Split data
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        # Train and evaluate models
        for model_name, train_func in [
            ("Random Forest", train_model),
            ("XGBoost", train_xgboost),
            ("LightGBM", train_lightgbm),
            ("Logistic Regression", train_logistic_regression)
        ]:
            logging.info(f"Training {model_name}...")
            model = train_func(X_train, y_train)

            # Evaluate with F1-score and additional metrics
            report = evaluate_model(model, X_test, y_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            log_loss_value = log_loss(y_test, y_pred_proba)

            # Log results for the current split and model
            logging.info(f"{model_name} Report:\n{report}")
            logging.info(f"{model_name} AUC-ROC: {auc:.4f}, Log-Loss: {log_loss_value:.4f}")

            # Append metrics to results
            results.append({
                "random_state": random_state,
                "model": model_name,
                "f1_score": report["weighted avg"]["f1-score"],
                "auc_roc": auc,
                "log_loss": log_loss_value,
            })

            # Save the model
            save_model(model, f"./models/{model_name.lower().replace(' ', '_')}_split_{random_state}.pkl")

            # Plot confusion matrix
            if show_plots:
                labels = ['No Diabetes (0)', 'Diabetes (1)']
                logging.info(f"Plotting confusion matrix for {model_name}...")
                plot_confusion_matrix(model, X_test, y_test, labels, model_name)

            # Plot feature importance (only for tree-based models)
            if show_plots and hasattr(model, 'feature_importances_'):
                logging.info(f"Plotting feature importance for {model_name}...")
                plot_feature_importance(model, X.columns, model_name)

    return results
