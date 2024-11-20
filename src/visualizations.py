import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_feature_importance(model, feature_names):
    """Plot feature importance from the trained model."""
    feature_importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importances)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.show()

def plot_confusion_matrix(model, X_test, y_test, labels):
    """
    Plot a confusion matrix for the given model and test data.
    
    Args:
        model: The trained model to evaluate.
        X_test: Test feature set.
        y_test: True labels for the test set.
        labels: List of labels (e.g., ['No Diabetes', 'Diabetes']).
    """
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()