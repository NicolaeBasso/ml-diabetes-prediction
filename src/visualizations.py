import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_feature_importance(model, feature_names, model_name):
    """
    Plot feature importance from the trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature names.
        model_name: Name of the model (e.g., "Random Forest").
    """
    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()  # Sort by importance

    # Create figure
    plt.figure(figsize=(12, 8))

    # Bar plot for feature importance
    plt.barh(
        [feature_names[i] for i in sorted_idx],
        feature_importances[sorted_idx],
        color='skyblue',
        edgecolor='black'
    )
    plt.xlabel("Feature Importance (Score)")
    plt.ylabel("Feature Names")
    plt.title(f"Feature Importance for {model_name} model")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Set the window title
    fig_manager = plt.get_current_fig_manager()
    fig_manager.set_window_title(f"{model_name} - Feature Importance")

    plt.show()

def plot_confusion_matrix(model, X_test, y_test, labels, model_name):
    """
    Plot a confusion matrix for the given model and test data.
    
    Args:
        model: The trained model to evaluate.
        X_test: Test feature set.
        y_test: True labels for the test set.
        labels: List of labels (e.g., ['No Diabetes', 'Diabetes']).
        model_name: Name of the model (e.g., "Random Forest").
    """
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create the plot using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
    
    # Add custom labels and title
    ax.set_title(f"Confusion Matrix for {model_name} model", fontsize=14)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Set the window title
    fig_manager = plt.get_current_fig_manager()
    fig_manager.set_window_title(f"{model_name} - Confusion Matrix")

    plt.show()