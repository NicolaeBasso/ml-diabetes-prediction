from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


def train_model(X_train, y_train, random_state=42):
    """Train the Random Forest Classifier."""
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return the classification report as a dictionary.
    
    Args:
        model: The trained model to evaluate.
        X_test: The test features.
        y_test: The true labels for the test set.
    
    Returns:
        dict: Classification report as a dictionary.
    """
    y_pred = model.predict(X_test)
    # Generate a classification report as a dictionary
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

def tune_hyperparameters_random(model, X_train, y_train, n_iter=20, cv=3):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.

    Args:
        model: The machine learning model to tune.
        X_train: Training feature set.
        y_train: Training labels.
        n_iter (int): Number of random combinations to try.
        cv (int): Number of cross-validation folds.

    Returns:
        tuple: Best model and best parameters.
    """
    param_dist = {
        'n_estimators': [20, 50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='f1_weighted',
        random_state=42,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier."""
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train):
    """Train a LightGBM classifier."""
    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def cross_validate_model(model, X, y, cv=5, scoring="f1_weighted"):
    """
    Perform cross-validation on the given model.

    Args:
        model: The machine learning model to evaluate.
        X: Feature matrix.
        y: Target labels.
        cv (int): Number of cross-validation folds (default=5).
        scoring (str): Metric for evaluation (default="f1_weighted").

    Returns:
        list: Cross-validation scores for each fold.
    """
    print(f"Performing {cv}-fold cross-validation...")
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return scores