from src.data_loader import load_data, split_data
from src.model import (
    train_model,
    evaluate_model,
    train_xgboost,
    train_lightgbm,
    train_logistic_regression,
    cross_validate_model,
    tune_hyperparameters_random
)
from src.utils import save_model
from src.visualizations import plot_feature_importance, plot_confusion_matrix

# File paths
DATA_PATH = "./data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
MODEL_PATH = "./models/diabetes_rf_model.pkl"

def main():
    # Step 1: Load and preprocess the data
    print("Loading data...")
    X, y = load_data(DATA_PATH)

    # Step 2: Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3: Train the baseline Random Forest model
    print("Training baseline Random Forest model...")
    model = train_model(X_train, y_train)

    # Step 4: Evaluate the model
    print("Evaluating baseline Random Forest model...")
    report = evaluate_model(model, X_test, y_test)
    print(report)

    # Step 5: Plot feature importance for baseline model
    print("Plotting feature importance for Random Forest...")
    plot_feature_importance(model, X.columns)

    # Step 6: Hyperparameter tuning for Random Forest
   # Perform random search hyperparameter tuning
    print("Tuning hyperparameters using RandomizedSearchCV...")
    best_model, best_params = tune_hyperparameters_random(model, X_train, y_train, n_iter=20)
    print(f"Best Parameters: {best_params}")

    # Step 7: Evaluate the tuned model
    print("Evaluating tuned Random Forest model...")
    tuned_report = evaluate_model(best_model, X_test, y_test)
    print(tuned_report)

    # Step 8: Save the best model
    print("Saving tuned Random Forest model...")
    save_model(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Step 9: Train and evaluate alternative models
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_report = evaluate_model(xgb_model, X_test, y_test)
    print("\nXGBoost Evaluation:")
    print(xgb_report)

    print("Training LightGBM...")
    lgb_model = train_lightgbm(X_train, y_train)
    lgb_report = evaluate_model(lgb_model, X_test, y_test)
    print("\nLightGBM Evaluation:")
    print(lgb_report)

    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_report = evaluate_model(lr_model, X_test, y_test)
    print("\nLogistic Regression Evaluation:")
    print(lr_report)    

    # Step 10: Plot confusion matrix for tuned Random Forest model
    print("Plotting confusion matrix for tuned Random Forest model...")
    labels = ['No Diabetes (0)', 'Diabetes (1)']
    plot_confusion_matrix(best_model, X_test, y_test, labels)

    # Step 11: Perform cross-validation
    print("Performing cross-validation on tuned Random Forest model...")
    cv_scores = cross_validate_model(best_model, X, y)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Average F1-Score: {cv_scores.mean():.4f}")

if __name__ == "__main__":
    main()
